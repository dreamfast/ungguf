#!/usr/bin/env bash
set -euo pipefail

CYAN='\033[36m'
BOLD='\033[1m'
RED='\033[31m'
RESET='\033[0m'

if [ -t 1 ]; then
	echo -e "${CYAN}"
	cat <<'BANNER'
                _ 
   ._  _  _   _|_ 
|_|| |(_|(_||_||  
       _| _|  v0.1.0
BANNER
	echo -e "${RESET}"
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_VLLM="ungguf-inference-vllm:latest"

# Export host UID/GID so Docker containers create files owned by the host user.
# NOTE: Do NOT use 'UID' — it is a readonly variable in bash.
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

die() {
	echo -e "${RED}Error: $*${RESET}" >&2
	exit 1
}

split_path() {
	local fullpath="$1"
	local dir fn
	dir="$(cd "$(dirname "$fullpath")" && pwd)"
	fn="$(basename "$fullpath")"
	echo "$dir" "$fn"
}

cmd_build() {
	local target="${1:-convert}"
	case "$target" in
	convert)
		docker compose -f "$REPO_DIR/docker-compose.yml" build
		;;
	inference)
		docker build \
			--build-arg HOST_UID="$HOST_UID" \
			--build-arg HOST_GID="$HOST_GID" \
			-f "$REPO_DIR/Dockerfile.inference" -t "$IMAGE_VLLM" "$REPO_DIR"
		;;
	*)
		die "Unknown build target: $target. Use 'convert' (default) or 'inference'."
		;;
	esac
}

_ensure_vllm_image() {
	if ! docker image inspect "$IMAGE_VLLM" &>/dev/null; then
		die "vLLM image ($IMAGE_VLLM) not found. Build it first:\n  docker build -f Dockerfile.inference -t $IMAGE_VLLM ."
	fi
}

cmd_convert_qwen35() {
	local keep_fp16=false
	if [[ "${1:-}" == "--keep-fp16" ]]; then
		keep_fp16=true
		shift
	fi
	[[ $# -ge 3 ]] || die "Usage: ungguf convert-qwen35 [--keep-fp16] <gguf_file> <output_dir> <ref_model_dir>"
	local gguf_path="$1"
	local output_dir="$2"
	local ref_dir="$3"

	local gguf_dir gguf_file
	read -r gguf_dir gguf_file <<<"$(split_path "$gguf_path")"
	output_dir="$(cd "$output_dir" && pwd)"
	ref_dir="$(cd "$ref_dir" && pwd)"

	export GPU="${GPU:-0}"
	export SHARD_SIZE_MB="${SHARD_SIZE_MB:-4500}"
	export GGUF_DIR="$gguf_dir"
	export GGUF_FILE="$gguf_file"
	export OUTPUT_DIR="$output_dir"
	export REF_MODEL_DIR="$ref_dir"

	local cmd=(python3 gguf_to_safetensors_qwen35.py
		--gguf "/input/$gguf_file" --output /output
		--reference-model /ref --shard-size-mb "${SHARD_SIZE_MB:-4500}")
	if $keep_fp16; then cmd+=(--keep-fp16); fi

	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm convert-qwen35 "${cmd[@]}"
}

cmd_convert_glm47() {
	local keep_fp16=false
	if [[ "${1:-}" == "--keep-fp16" ]]; then
		keep_fp16=true
		shift
	fi
	[[ $# -ge 3 ]] || die "Usage: ungguf convert-glm47 [--keep-fp16] <gguf_file> <output_dir> <ref_model_dir>"
	local gguf_path="$1"
	local output_dir="$2"
	local ref_dir="$3"

	local gguf_dir gguf_file
	read -r gguf_dir gguf_file <<<"$(split_path "$gguf_path")"
	output_dir="$(cd "$output_dir" && pwd)"
	ref_dir="$(cd "$ref_dir" && pwd)"

	export GPU="${GPU:-0}"
	export SHARD_SIZE_MB="${SHARD_SIZE_MB:-4500}"
	export GGUF_DIR="$gguf_dir"
	export GGUF_FILE="$gguf_file"
	export OUTPUT_DIR="$output_dir"
	export REF_MODEL_DIR="$ref_dir"

	local cmd=(python3 gguf_to_safetensors_glm47.py
		--gguf "/input/$gguf_file" --output /output
		--reference-model /ref --shard-size-mb "${SHARD_SIZE_MB:-4500}")
	if $keep_fp16; then cmd+=(--keep-fp16); fi

	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm convert-glm47 "${cmd[@]}"
}

cmd_convert_qwen3() {
	local keep_fp16=false
	if [[ "${1:-}" == "--keep-fp16" ]]; then
		keep_fp16=true
		shift
	fi
	[[ $# -ge 3 ]] || die "Usage: ungguf convert-qwen3 [--keep-fp16] <gguf_file> <output_dir> <ref_model_dir>"
	local gguf_path="$1"
	local output_dir="$2"
	local ref_dir="$3"

	local gguf_dir gguf_file
	read -r gguf_dir gguf_file <<<"$(split_path "$gguf_path")"
	output_dir="$(cd "$output_dir" && pwd)"
	ref_dir="$(cd "$ref_dir" && pwd)"

	export GPU="${GPU:-0}"
	export SHARD_SIZE_MB="${SHARD_SIZE_MB:-4500}"
	export GGUF_DIR="$gguf_dir"
	export GGUF_FILE="$gguf_file"
	export OUTPUT_DIR="$output_dir"
	export REF_MODEL_DIR="$ref_dir"

	local cmd=(python3 gguf_to_safetensors_qwen3.py
		--gguf "/input/$gguf_file" --output /output
		--reference-model /ref
		--shard-size-mb "${SHARD_SIZE_MB:-4500}")
	if $keep_fp16; then cmd+=(--keep-fp16); fi

	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm convert-qwen3 "${cmd[@]}"
}

cmd_verify() {
	local keep_fp16=false
	if [[ "${1:-}" == "--keep-fp16" ]]; then
		keep_fp16=true
		shift
	fi
	[[ $# -ge 3 ]] || die "Usage: ungguf verify [--keep-fp16] <gguf_file> <converted_dir> <ref_model_dir> [results_dir]"
	local gguf_path="$1"
	local converted_dir="$2"
	local ref_dir="$3"
	local results_dir="${4:-$(pwd)/results}"

	local gguf_dir gguf_file
	read -r gguf_dir gguf_file <<<"$(split_path "$gguf_path")"
	converted_dir="$(cd "$converted_dir" && pwd)"
	ref_dir="$(cd "$ref_dir" && pwd)"
	mkdir -p "$results_dir"
	results_dir="$(cd "$results_dir" && pwd)"

	export GPU="${GPU:-0}"
	export GGUF_DIR="$gguf_dir"
	export GGUF_FILE="$gguf_file"
	export OUTPUT_DIR="$converted_dir"
	export REF_MODEL_DIR="$ref_dir"
	export RESULTS_DIR="$results_dir"

	local cmd=(python3 verify_conversion_qwen35.py
		--gguf "/input/$gguf_file" --converted /converted
		--reference /ref --output /results/conversion_verification.json)
	if $keep_fp16; then cmd+=(--keep-fp16); fi

	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm verify "${cmd[@]}"
}

cmd_verify_glm47() {
	local keep_fp16=false
	if [[ "${1:-}" == "--keep-fp16" ]]; then
		keep_fp16=true
		shift
	fi
	[[ $# -ge 3 ]] || die "Usage: ungguf verify-glm47 [--keep-fp16] <gguf_file> <converted_dir> <ref_model_dir> [results_dir]"
	local gguf_path="$1"
	local converted_dir="$2"
	local ref_dir="$3"
	local results_dir="${4:-$(pwd)/results}"

	local gguf_dir gguf_file
	read -r gguf_dir gguf_file <<<"$(split_path "$gguf_path")"
	converted_dir="$(cd "$converted_dir" && pwd)"
	ref_dir="$(cd "$ref_dir" && pwd)"
	mkdir -p "$results_dir"
	results_dir="$(cd "$results_dir" && pwd)"

	export GPU="${GPU:-0}"
	export GGUF_DIR="$gguf_dir"
	export GGUF_FILE="$gguf_file"
	export OUTPUT_DIR="$converted_dir"
	export REF_MODEL_DIR="$ref_dir"
	export RESULTS_DIR="$results_dir"

	local cmd=(python3 verify_conversion_glm47.py
		--gguf "/input/$gguf_file" --converted /converted
		--reference /ref --output /results/glm47_verification.json)
	if $keep_fp16; then cmd+=(--keep-fp16); fi

	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm verify-glm47 "${cmd[@]}"
}

cmd_verify_qwen3() {
	[[ $# -ge 2 ]] || die "Usage: ungguf verify-qwen3 <gguf_file> <converted_dir>"
	local gguf_path="$1"
	local converted_dir="$2"

	local gguf_dir gguf_file
	read -r gguf_dir gguf_file <<<"$(split_path "$gguf_path")"
	converted_dir="$(cd "$converted_dir" && pwd)"

	export GPU="${GPU:-0}"
	export GGUF_DIR="$gguf_dir"
	export GGUF_FILE="$gguf_file"
	export OUTPUT_DIR="$converted_dir"

	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm verify-qwen3 \
		python3 verify_conversion_qwen3.py \
		--gguf "/input/$gguf_file" --converted /converted
}

cmd_inspect() {
	[[ $# -ge 1 ]] || die "Usage: ungguf inspect <gguf_file> [gguf_file2 ...]"
	local first_file="$1"
	shift

	local first_dir first_fn
	read -r first_dir first_fn <<<"$(split_path "$first_file")"

	local filenames=("$first_fn")
	for f in "$@"; do
		local fdir fn
		read -r fdir fn <<<"$(split_path "$f")"
		[[ "$fdir" == "$first_dir" ]] || die "All files must be in the same directory. Got $first_dir and $fdir"
		filenames+=("$fn")
	done

	local -a file_args=()
	for fn in "${filenames[@]}"; do
		file_args+=("/input/$fn")
	done

	export GPU="${GPU:-0}"
	export GGUF_DIR="$first_dir"
	export GGUF_FILE="${filenames[0]}"

	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm \
		inspect \
		python3 /app/gguf_metadata_dump.py "${file_args[@]}"
}

cmd_sanity() {
	[[ $# -ge 1 ]] || die "Usage: ungguf sanity <model_dir_or_gguf> [--label X] [--quantize fp8] [--tp N] [--tokenizer /path] [--max-model-len N]"
	local model_path="$1"
	shift

	local label="sanity"
	local quantize=""
	local tp=""
	local tokenizer=""
	local max_model_len=""
	while [[ $# -gt 0 ]]; do
		case "$1" in
		--label)
			label="$2"
			shift 2
			;;
		--quantize)
			quantize="$2"
			shift 2
			;;
		--tp)
			tp="$2"
			shift 2
			;;
		--tokenizer)
			tokenizer="$2"
			shift 2
			;;
		--max-model-len)
			max_model_len="$2"
			shift 2
			;;
		*) die "Unknown argument: $1" ;;
		esac
	done

	local gpu="${GPU:-all}"
	local results_dir
	results_dir="$(pwd)/results"
	mkdir -p "$results_dir"
	results_dir="$(cd "$results_dir" && pwd)"

	_ensure_vllm_image

	# Determine if model is a GGUF file or safetensors directory
	local volumes=()
	local model_arg
	if [[ "$model_path" == *.gguf ]]; then
		local gguf_dir gguf_file
		read -r gguf_dir gguf_file <<<"$(split_path "$model_path")"
		volumes+=(-v "$gguf_dir:/input:ro")
		model_arg="/input/$gguf_file"
	else
		model_path="$(cd "$model_path" && pwd)"
		volumes+=(-v "$model_path:/model:ro")
		model_arg="/model"
	fi

	# Optional tokenizer mount
	local tok_args=()
	if [[ -n "$tokenizer" ]]; then
		tokenizer="$(cd "$tokenizer" && pwd)"
		volumes+=(-v "$tokenizer:/tokenizer:ro")
		tok_args=(--tokenizer /tokenizer)
	fi

	local extra_args=()
	[[ -n "$quantize" ]] && extra_args+=(--quantize "$quantize")
	[[ -n "$tp" ]] && extra_args+=(--tp "$tp")
	[[ -n "$max_model_len" ]] && extra_args+=(--max-model-len "$max_model_len")

	echo -e "${BOLD}Running sanity check (GPU=${gpu}${quantize:+, quantize=$quantize}${tp:+, tp=$tp})${RESET}"
	docker run --rm --runtime=nvidia \
		--user "${HOST_UID}:${HOST_GID}" \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-e NVIDIA_VISIBLE_DEVICES="$gpu" \
		-e CUDA_VISIBLE_DEVICES="$gpu" \
		-e PYTHONUNBUFFERED=1 \
		-e PYTORCH_ALLOC_CONF=expandable_segments:True \
		-e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}" \
		-e VLLM_WORKER_MULTIPROC_METHOD=spawn \
		"${volumes[@]}" \
		-v "$results_dir:/results" \
		"$IMAGE_VLLM" \
		python3 vllm_sanity.py \
		--model "$model_arg" \
		--label "$label" \
		--output "/results/${label}.json" \
		"${tok_args[@]}" \
		"${extra_args[@]}"
}

cmd_test() {
	echo -e "${BOLD}Running pytest inside Docker (CPU-only)${RESET}"
	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm test \
		python3 -m pytest /app/tests/ -v --tb=short "$@"
}

cmd_lint() {
	echo -e "${BOLD}Running ruff check + ruff format --check + mypy inside Docker${RESET}"
	docker compose -f "$REPO_DIR/docker-compose.yml" run --rm lint
}

cmd_format() {
	echo -e "${BOLD}Running ruff format + ruff check --fix inside Docker${RESET}"
	# Cannot use 'docker compose run' here because compose overlays -v on top
	# of the :ro mounts defined in docker-compose.yml instead of replacing them.
	# Use 'docker run' directly with writable mounts.
	docker run --rm \
		--user "${HOST_UID}:${HOST_GID}" \
		-v "$REPO_DIR/src:/app/src" \
		-v "$REPO_DIR/tests:/app/tests" \
		-v "$REPO_DIR/pyproject.toml:/app/pyproject.toml:ro" \
		-v "$REPO_DIR/gguf_metadata_dump.py:/app/gguf_metadata_dump.py" \
		-e RUFF_CACHE_DIR=/tmp/.ruff_cache \
		-w /app \
		ungguf-lint \
		sh -c "ruff check --fix . && ruff format ."
}

cmd_help() {
	echo -e "Usage: ${BOLD}ungguf${RESET} <command> [args...]"
	echo ""
	echo -e "  ${BOLD}build${RESET} [convert|inference]   Build Docker images (default: convert)"
	echo -e "  ${BOLD}convert-qwen35${RESET} [--keep-fp16] <g> <o> <r>  Convert Qwen3.5 GGUF to safetensors"
	echo -e "  ${BOLD}convert-glm47${RESET} [--keep-fp16] <g> <o> <r>  Convert GLM-4.7 / deepseek2 GGUF"
	echo -e "  ${BOLD}convert-qwen3${RESET} [--keep-fp16] <g> <o> <r>   Convert Qwen3 GGUF to safetensors"
	echo -e "  ${BOLD}verify${RESET} [--keep-fp16] <g> <c> <ref>   Verify Qwen3.5 conversion is bit-exact"
	echo -e "  ${BOLD}verify-glm47${RESET} [--keep-fp16] <g> <c> <ref>  Verify GLM-4.7 conversion is bit-exact"
	echo -e "  ${BOLD}verify-qwen3${RESET} <gguf> <converted_dir>       Verify Qwen3 conversion is bit-exact"
	echo -e "  ${BOLD}inspect${RESET} <gguf> [gguf2...]     Dump GGUF metadata and tensor names"
	echo -e "  ${BOLD}sanity${RESET} <model|gguf> [opts]    Run vLLM inference sanity check (GGUF or safetensors)"
	echo -e "         [--label X] [--quantize fp8] [--tp N] [--tokenizer /path] [--max-model-len N]"
	echo -e "  ${BOLD}test${RESET} [pytest-args...]          Run unit tests inside Docker (CPU-only)"
	echo -e "  ${BOLD}lint${RESET}                           Run ruff + mypy checks inside Docker (read-only)"
	echo -e "  ${BOLD}format${RESET}                         Auto-fix lint issues and format code via Docker"
	echo -e "  ${BOLD}help${RESET}                          Show this help message"
	echo ""
	echo "Environment variables:"
	echo "  GPU             GPU device(s) — default: 'all' for sanity, '0' for convert/verify"
	echo "                  Examples: GPU=0 GPU=1 GPU=0,1 GPU=all"
	echo "  SHARD_SIZE_MB   Shard size in MB for convert (default: 4500)"
}

case "${1:-help}" in
build)
	shift
	cmd_build "$@"
	;;
convert-qwen35)
	shift
	cmd_convert_qwen35 "$@"
	;;
convert-glm47)
	shift
	cmd_convert_glm47 "$@"
	;;
convert-qwen3)
	shift
	cmd_convert_qwen3 "$@"
	;;
verify)
	shift
	cmd_verify "$@"
	;;
verify-glm47)
	shift
	cmd_verify_glm47 "$@"
	;;
verify-qwen3)
	shift
	cmd_verify_qwen3 "$@"
	;;
inspect)
	shift
	cmd_inspect "$@"
	;;
sanity)
	shift
	cmd_sanity "$@"
	;;
test)
	shift
	cmd_test "$@"
	;;
lint)
	shift
	cmd_lint "$@"
	;;
format)
	shift
	cmd_format "$@"
	;;
help | --help | -h) cmd_help ;;
*) die "Unknown command: $1. Run 'ungguf help' for usage." ;;
esac
