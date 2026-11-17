#!/usr/bin/env python3
"""Run an Iludev GGUF with prebuilt llama.cpp (no compiling, no pip packages).

	python run_gguf.py iludev-q8.gguf              # chat in the terminal
	python run_gguf.py iludev-q8.gguf -p "Hello"   # one answer, then exit

The first run downloads the latest prebuilt llama.cpp release for your OS into ./llama-bin.
In chat: /reset clears the conversation, /quit exits.
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

UA = {"User-Agent": "iludev-run-gguf"}


def platform_keys():
	s, m = platform.system(), platform.machine().lower()
	arm = m in ("arm64", "aarch64")
	if s == "Linux":
		return ["ubuntu-arm64" if arm else "ubuntu-x64"]
	if s == "Darwin":
		return ["macos-arm64" if arm else "macos-x64"]
	if s == "Windows":
		return ["win-cpu-x64", "win-avx2-x64"]
	sys.exit(f"Unsupported system: {s} {m}")


def find_server(root):
	name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
	for d, _, files in os.walk(root):
		if name in files:
			return os.path.join(d, name)
	return None


def fetch(url):
	return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)


def find_asset():
	"""Newest release that already has a build for this OS (the very latest one is sometimes still uploading)."""
	feed = fetch("https://github.com/ggml-org/llama.cpp/releases.atom").read().decode()
	tags = []
	for t in re.findall(r"/releases/tag/(b\d+)", feed):
		if t not in tags:
			tags.append(t)
	for tag in tags[:12]:
		page = fetch(f"https://github.com/ggml-org/llama.cpp/releases/expanded_assets/{tag}").read().decode()
		for key in platform_keys():
			m = re.search(rf"releases/download/{tag}/(llama-{tag}-bin-{re.escape(key)}\.(?:zip|tar\.gz))\"", page)
			if m:
				return tag, m.group(1)
	sys.exit("No prebuilt llama.cpp build found for this system: https://github.com/ggml-org/llama.cpp/releases")


def download(root):
	tag, name = find_asset()
	url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
	pkg = os.path.join(root, name)
	print("Downloading", url)
	with fetch(url) as r, open(pkg, "wb") as f:
		while True:
			chunk = r.read(1 << 20)
			if not chunk:
				break
			f.write(chunk)
	return pkg


def ensure_llama(root):
	exe = find_server(root)
	if exe:
		return exe
	os.makedirs(root, exist_ok=True)
	pkg = download(root)
	if pkg.endswith(".zip"):
		zipfile.ZipFile(pkg).extractall(root)
	else:
		tarfile.open(pkg).extractall(root)
	os.remove(pkg)
	exe = find_server(root)
	if not exe:
		sys.exit("Downloaded the release but could not find llama-server in it.")
	if platform.system() != "Windows":
		for d, _, files in os.walk(os.path.dirname(exe)):
			for fn in files:
				p = os.path.join(d, fn)
				if "." not in fn:
					os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
	return exe


def ensure_libgomp(libdir):
	"""The prebuilt Linux llama.cpp needs the OpenMP runtime (libgomp). Install it or borrow a copy if it is missing."""
	if platform.system() != "Linux":
		return
	try:
		if "libgomp.so.1" in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout:
			return
	except Exception:
		pass
	if os.path.exists(os.path.join(libdir, "libgomp.so.1")):
		return
	print("libgomp is missing, trying to install it...")
	cmds = []
	if shutil.which("apt-get"):
		if os.geteuid() == 0:
			cmds.append("apt-get update -qq && apt-get install -y -qq libgomp1")
		elif shutil.which("sudo"):
			cmds.append("sudo apt-get update -qq && sudo apt-get install -y -qq libgomp1")
	for c in cmds:
		if subprocess.run(c, shell=True).returncode == 0:
			return
	for base in sys.path:  # a copy shipped inside pip packages such as torch or scikit-learn
		for pat in ("torch/lib/libgomp*", "*.libs/libgomp*"):
			for hit in glob.glob(os.path.join(base, pat)):
				os.symlink(hit, os.path.join(libdir, "libgomp.so.1"))
				print("using", hit)
				return
	print("Could not install libgomp automatically. Run: sudo apt-get install libgomp1")


def free_port():
	with socket.socket() as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def post(port, payload):
	req = urllib.request.Request(
		f"http://127.0.0.1:{port}/v1/chat/completions",
		data=json.dumps(payload).encode(),
		headers={"Content-Type": "application/json"},
	)
	return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip()


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("gguf", help="path to the .gguf file")
	ap.add_argument("-p", "--prompt", default="", help="ask one question and exit")
	ap.add_argument("--bin-dir", default="llama-bin", help="where prebuilt llama.cpp is kept")
	ap.add_argument("--ctx", type=int, default=1024)
	ap.add_argument("--max", type=int, default=120, help="max tokens per answer")
	ap.add_argument("--temp", type=float, default=0.7)
	ap.add_argument("--min-p", type=float, default=0.05)
	ap.add_argument("--freq", type=float, default=0.3, help="frequency penalty")
	ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
	args = ap.parse_args()
	if not os.path.exists(args.gguf):
		sys.exit("GGUF not found: " + args.gguf)

	exe = ensure_llama(args.bin_dir)
	port = free_port()
	env = dict(os.environ)
	libdir = os.path.dirname(exe)
	ensure_libgomp(libdir)
	env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
	env["DYLD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
	logf = open("llama-server.log", "w")
	proc = subprocess.Popen(
		[exe, "-m", args.gguf, "--host", "127.0.0.1", "--port", str(port),
		 "-c", str(args.ctx), "-t", str(args.threads)],
		stdout=logf, stderr=subprocess.STDOUT, env=env,
	)
	try:
		print("Loading the model...")
		for _ in range(120):
			if proc.poll() is not None:
				logf.flush()
				print(open("llama-server.log").read()[-1500:])
				sys.exit("llama-server stopped. If the log says the architecture is unknown, "
						 "delete ./" + args.bin_dir + " and run again to get a newer release.")
			try:
				urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
				break
			except Exception:
				time.sleep(1)
		else:
			sys.exit("Timed out waiting for the model to load. See llama-server.log")

		def ask(history, q):
			history.append({"role": "user", "content": q})
			a = post(port, {"messages": history, "max_tokens": args.max, "temperature": args.temp,
							"min_p": args.min_p, "frequency_penalty": args.freq})
			history.append({"role": "assistant", "content": a})
			return a

		history = []
		if args.prompt:
			print(ask(history, args.prompt))
			return
		print("Ready. /reset clears the chat, /quit exits.")
		while True:
			try:
				q = input("you> ").strip()
			except (EOFError, KeyboardInterrupt):
				break
			if not q:
				continue
			if q in ("/quit", "/exit", "exit"):
				break
			if q in ("/reset", "/clear"):
				history = []
				print("(cleared)")
				continue
			print("iludev>", ask(history, q))
	finally:
		proc.terminate()
		try:
			proc.wait(timeout=5)
		except Exception:
			proc.kill()


if __name__ == "__main__":
	main()
