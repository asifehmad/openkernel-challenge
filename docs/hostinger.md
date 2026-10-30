# Official CLI setup on the Hostinger VPS

These are commands for the owner's Linux VPS, **not commands already executed
there**. The agent has no attached VPS connection. Its cloud proxy rejected the
official domain before connecting to the origin; repeating requests through that
same proxy will not resolve it. This workflow uses the VPS's normal authorized
network route, with TLS verification intact.

The current official installer, CLI version, authentication requirements and
task bundle are still unverified. A historical public starter documents
`https://openkernelchallenge.dev/install.sh` and `$HOME/.okc/bin`; confirm the
installer's current contents and actual destination before executing it.
Never install the unrelated PyPI package named `okc`.

## 1. Download the installer and official pages

Connect to your VPS using your existing SSH access, then run the following as
your normal user. On a Debian/Ubuntu VPS, install any missing basic tools first:

```bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3 less ripgrep
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Use Python 3.10+ (Ubuntu 22.04/24.04 or newer)"'

umask 077
okc_stage="$(mktemp -d "$HOME/okc-bootstrap.XXXXXX")"
export okc_stage
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$okc_stage/retrieved-utc.txt"
curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
  --connect-timeout 15 --max-time 120 \
  https://openkernelchallenge.dev/install.sh \
  -D "$okc_stage/installer-headers.txt" -o "$okc_stage/install.sh"
sha256sum "$okc_stage/install.sh" > "$okc_stage/installer-sha256.txt"
less "$okc_stage/install.sh"
```

Read the installer **before proceeding to the next block**. Check its platform
selection, installation directory, artifact download URLs and any checksum or
signature validation. Preserve its verification; do not disable TLS or change
expected hashes. The locally calculated installer hash records the bytes you
reviewed, not an independent signature verification.

If this request also fails on the VPS, stop and retain the HTTP status and
non-sensitive error. Fix the VPS/provider egress or DNS configuration through
its normal controls; do not use an unrelated relay to evade an access policy.
Do not execute an error page or substitute a third-party CLI binary.

Once the installer review succeeds, record the official pages for comparison:

```bash
curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
  https://openkernelchallenge.dev/ -o "$okc_stage/home.html"
curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
  https://openkernelchallenge.dev/rules -o "$okc_stage/rules.html"
curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
  https://openkernelchallenge.dev/problems/muse_bf16_proj -o "$okc_stage/problem.html"
sha256sum "$okc_stage/"*.html > "$okc_stage/pages-sha256.txt"
```

Read the live pages in your browser as well. An HTML hash identifies downloaded
bytes; it does not prove that dynamically loaded leaderboard/rules data was
captured or that the site's rules have a particular version identifier.

## 2. Install the reviewed CLI and identify its version

If the inspected installer prescribes execution as a shell script, run that
reviewed file. Do not add guessed flags or run it as root unless the official
instructions require a system installation you have deliberately chosen.

```bash
bash "$okc_stage/install.sh"
# Use this PATH entry only if the installer confirmed this destination:
export PATH="$HOME/.okc/bin:$PATH"
command -v okc
okc --help > "$okc_stage/okc-help.txt"
cat "$okc_stage/okc-help.txt"
```

Record the version using the option/subcommand actually advertised by this
installed CLI. The following selects two common forms only when help lists
them; it stops instead of inventing a version when neither is documented:

```bash
if rg -q -- '--version' "$okc_stage/okc-help.txt"; then
  okc --version > "$okc_stage/okc-version.txt"
elif rg -q '^[[:space:]]+version([[:space:]]|$)' "$okc_stage/okc-help.txt"; then
  okc version > "$okc_stage/okc-version.txt"
else
  echo 'No recognized version command in help. Use the CLI-documented mechanism and record its actual output before proceeding.' >&2
  exit 1
fi
test -s "$okc_stage/okc-version.txt"
cat "$okc_stage/okc-version.txt"
okc_binary="$(command -v okc)"
test -f "$okc_binary"
sha256sum "$okc_binary" > "$okc_stage/cli-sha256.txt"
okc init --help > "$okc_stage/init-help.txt"
okc problems --help > "$okc_stage/problems-help.txt"
okc leaderboard --help > "$okc_stage/leaderboard-help.txt"
cat "$okc_stage/init-help.txt" "$okc_stage/problems-help.txt" "$okc_stage/leaderboard-help.txt"
```

Confirm the documented argument forms support the commands below. If they have
changed, use the documented syntax and retain that help output. Do not claim a
CLI version until the installed executable actually reports it.

## 3. Handle authentication only if required

Try the documented read-only commands first:

```bash
okc problems
okc leaderboard muse_bf16_proj
```

If they require authentication, **pause automation and let the owner complete
this step interactively**:

```bash
okc login --help
okc login
```

Complete the browser/GitHub authorization and personally review any competition
terms. An agent must not accept terms for you. Then retry the read-only commands
and record their output locally:

```bash
okc problems > "$okc_stage/problems.txt"
okc leaderboard muse_bf16_proj > "$okc_stage/leaderboard.txt"
```

`okc problems` is where the installed CLI should expose the current limits;
read the actual output rather than assuming earlier quotas still apply. Do not
run `okc check` or `okc submit`: each requires separate explicit approval.
Never paste token values into chat, command arguments, repository files or logs.
Do not enable shell tracing or copy the CLI's authentication directory.

## 4. Fetch a new bundle and compare it without overwriting work

Use the existing feature branch. For a new VPS checkout:

```bash
git clone --branch feat/muse-bf16-proj https://github.com/asifehmad/openkernel-challenge.git
cd openkernel-challenge
git status --short --branch
git rev-parse HEAD
```

For an existing checkout, inspect its status first and preserve local edits;
do not reset it to make setup work. The following fetch creates a new ignored
directory. A prior bundle is never overwritten:

```bash
okc_bundle="official/vps-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/fetch_official.sh "$okc_bundle"
export OFFICIAL_DIR="$(python3 scripts/resolve_task.py "$okc_bundle")"
python3 scripts/compare_official.py --official-dir "$OFFICIAL_DIR" \
  --cli-version "$(cat "$okc_stage/okc-version.txt")" \
  --report "$okc_stage/official-comparison.json"
```

The directory is discovered from the output, not assumed to be
`official/muse_bf16_proj`. The comparison hashes TASK, AGENTS, baseline,
reference, checks and the starter submission. Differences in the starter
submission are expected; changes in rules, interfaces, shapes, dtype/tolerance
checks, baseline or scoring require review before GPU tests. Read both versions
and the current rules. Preserve the organizer's new files unchanged.

The report is a local file comparison, **not an official `okc check` result**.
Useful non-secret evidence to return is the CLI version, six-file comparison,
current rules/limits, and leaderboard output. Review files before sharing;
keep authentication material private.

## 5. Move only the task bundle to the rented H200

Clone the same feature branch on the H200 host. Copy only the downloaded task
directory to its ignored `official/` tree using your existing authorized SSH
connection. Do not copy `$HOME/.okc`, tokens, private keys or installer caches.
For example, replace the SSH host alias and repository path with your actual
authorized H200 destination:

```bash
scp -r "$OFFICIAL_DIR" h200-host:/path/to/openkernel-challenge/official/current
```

Create the destination's `official/` directory first, and refuse to overwrite an
existing `current` bundle. Compare the copied file hashes with the VPS record.
The H200 local runner does not need your competition token; it imports the
downloaded reference and baseline files locally.

Confirm the current official task still names
`lmsysorg/sglang:dev-muse-spark-open`. On the authorized GPU host, pull it once,
record its immutable digest, then start by that digest:

```bash
docker pull lmsysorg/sglang:dev-muse-spark-open
okc_image="$(docker image inspect lmsysorg/sglang:dev-muse-spark-open --format '{{index .RepoDigests 0}}')"
test -n "$okc_image"
printf '%s\n' "$okc_image" > benchmarks/results/evaluation-image.txt
docker run --rm -it --gpus all --ipc=host \
  --mount type=bind,src="$PWD",dst=/work --workdir /work "$okc_image" bash
```

Continue with [the H200 execution guide](h200.md), starting with environment and
single-configuration smoke validation before the full suite. Record the actual
image digest and dependency versions; the mutable tag alone does not establish
reproducibility. No GPU rental, quota use or competition submission is performed
by these instructions automatically.
