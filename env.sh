# common setup for both bash files

: "${VIAM_SIM_SANDBOX:=0}"

if [ $VIAM_SIM_SANDBOX = 1 ]; then
	export UV_PROJECT_ENVIRONMENT=/opt/isaac-sim-venv
else
	# we do this on non-sandbox machines because the shader cache is in the venv and
	# we want it to survive hot reloads.
	export UV_PROJECT_ENVIRONMENT="$VIAM_MODULE_DATA/isaac-venv-run2"
fi

export PATH="$PATH:$HOME/.local/bin"
if ! command -v uv >& /dev/null; then
	# todo: better way to manage min version of uv here
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi
