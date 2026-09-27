# PufferRoyale build / test entry points. All paths are relative to this directory
# (the project path contains spaces, so nothing here uses absolute paths).
#
#   make gen      regenerate pr_card_db.h / pr_arena_db.h from data/source (deterministic)
#   make build    build the pufferroyale.binding extension in place (-O2)
#   make debug    build the extension with -O0 -g -fsanitize=address,undefined into build/debug/
#                 (never over the in-place release extension)
#   make asan-py  the Python suites (builder + spec, minus subprocess / timing tests) against the
#                 build/debug extension under ASan + UBSan
#   make test-c   compile + run the C unit tests (clang -Wall -Wextra -Werror)
#   make asan     the C unit tests under AddressSanitizer + UBSan
#   make bench    C micro-benchmark: engine ticks/s
#   make bench-stress  worst cases: 216 Skeletons fighting across the river (audit C.1) and the
#                 SPEC §16 large battle
#   make test     test-c + pytest
#   make clean

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
ifeq ($(origin CC),default)
CC := clang
endif
CSRC := pufferroyale/csrc
BUILD := build/c
CWARN := -std=c99 -Wall -Wextra -Werror -Wshadow
CTESTS := $(patsubst tests/c/%.c,%,$(filter-out tests/c/bench.c,$(wildcard tests/c/test_*.c)))
HEADERS := $(wildcard $(CSRC)/*.h) tests/c/pr_test.h

.PHONY: all gen gen-check build debug test-c asan asan-py bench bench-stress test pytest clean

all: build

gen:
	$(PYTHON) tools/gen_card_db.py
	$(PYTHON) tools/gen_arena.py

gen-check:
	$(PYTHON) tools/gen_card_db.py --check
	$(PYTHON) tools/gen_arena.py --check

build:
	$(PYTHON) setup.py build_ext --inplace

debug:
	DEBUG=1 $(PYTHON) setup.py build_ext --force --build-lib build/debug --build-temp build/debug-tmp

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%: tests/c/%.c $(HEADERS) | $(BUILD)
	$(CC) $(CWARN) -O2 -I$(CSRC) -Itests/c $< -o $@

$(BUILD)/asan_%: tests/c/%.c $(HEADERS) | $(BUILD)
	$(CC) $(CWARN) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
		-fno-sanitize-recover=undefined -I$(CSRC) -Itests/c $< -o $@

# Every test binary runs under a hard wall-clock limit (perl alarm: SIGALRM after
# TEST_TIMEOUT s), so a hang can never stall a run. ASan reports are not symbolized by default
# (symbolizing through atos can hang in sandboxed macOS sessions); ASAN_SYMBOLIZE=1 turns it on.
TEST_TIMEOUT ?= 300
ASAN_SYMBOLIZE ?= 0
WITH_TIMEOUT := perl -e 'alarm shift @ARGV; exec @ARGV or die "exec: $$!"' $(TEST_TIMEOUT)
ASAN_ENV := ASAN_OPTIONS=detect_leaks=0:symbolize=$(ASAN_SYMBOLIZE) UBSAN_OPTIONS=print_stacktrace=1

test-c: $(addprefix $(BUILD)/,$(CTESTS))
	@set -e; for t in $(CTESTS); do echo "== $$t"; $(WITH_TIMEOUT) ./$(BUILD)/$$t; done

asan: $(addprefix $(BUILD)/asan_,$(CTESTS))
	@set -e; for t in $(CTESTS); do echo "== asan $$t"; $(ASAN_ENV) $(WITH_TIMEOUT) ./$(BUILD)/asan_$$t; done

$(BUILD)/bench: tests/c/bench.c $(HEADERS) | $(BUILD)
	$(CC) $(CWARN) -O2 -I$(CSRC) -Itests/c $< -o $@

bench: $(BUILD)/bench
	./$(BUILD)/bench

$(BUILD)/bench_stress: tests/c/bench_stress.c $(HEADERS) | $(BUILD)
	$(CC) $(CWARN) -O2 -I$(CSRC) -Itests/c $< -o $@

bench-stress: $(BUILD)/bench_stress
	./$(BUILD)/bench_stress

pytest:
	$(PYTHON) -m pytest -q

# The sanitizer runtime must be loaded first: DYLD_INSERT_LIBRARIES on macOS, LD_PRELOAD on Linux.
# Child processes (cross-process / subprocess tests) cannot inherit it reliably, so those tests
# and the timing tests are left out here (they run against the release build in `make pytest`).
ASAN_RT := $(shell $(CC) -print-file-name=$(if $(filter Darwin,$(shell uname -s)),libclang_rt.asan_osx_dynamic.dylib,libasan.so))
PRELOAD := $(if $(filter Darwin,$(shell uname -s)),DYLD_INSERT_LIBRARIES,LD_PRELOAD)
# Phase F / second-audit tests that run the scripts (or a fuzz) in child processes (node-id prefixes)
ASAN_PY_SKIP_SUBPROC := \
	--deselect tests/builder/test_audit2_training.py::test_best_response_accepts_run_dirs_ckpt_and_comma_decks \
	--deselect tests/builder/test_audit2_training.py::test_enabling_mmd_at_resume_uses_the_loaded_learner \
	--deselect tests/builder/test_audit2_training.py::test_failed_resume_exits_promptly \
	--deselect tests/builder/test_audit2_training.py::test_nan_epoch_is_never_saved \
	--deselect tests/builder/test_audit2_training.py::test_resume_applies_cli_optimizer_settings_and_restores_state \
	--deselect tests/builder/test_audit2_training.py::test_run_dir_is_relocatable \
	--deselect tests/builder/test_audit2_training.py::test_total_timesteps_below_one_batch_is_a_clear_error \
	--deselect tests/builder/test_audit2_training.py::test_tournament_and_eval_report_training_settings \
	--deselect tests/builder/test_llm_phase_f.py::test_llm_match_script \
	--deselect tests/spec/test_audit_v041.py::test_best_response_accepts_card_lists_and_run_dir_targets \
	--deselect tests/spec/test_audit_v041.py::test_league_enabling_mmd_at_resume_starts_from_the_learner \
	--deselect tests/spec/test_audit_v041.py::test_league_resume_of_corrupted_run_fails_promptly \
	--deselect tests/spec/test_audit_v041.py::test_league_resume_reapplies_cli_learning_rate \
	--deselect tests/spec/test_audit_v041.py::test_league_run_directory_is_relocatable \
	--deselect tests/spec/test_audit_v041.py::test_script_json_outputs_are_strict_json \
	--deselect tests/spec/test_audit_v041.py::test_snapshot_validation_bounds_multiplied_values_and_pending_spawns \
	--deselect tests/spec/test_hpc_templates.py::test_sbatch_scripts_and_flags_exist \
	--deselect tests/spec/test_league.py::test_league_train_cli_flags \
	--deselect tests/spec/test_llm.py::test_anthropic_adapter_without_sdk_raises_a_clear_error \
	--deselect tests/spec/test_llm.py::test_llm_match_mock_run_summary_and_transcripts \
	--deselect tests/spec/test_llm.py::test_llm_match_refuses_anthropic_without_allow_network \
	--deselect tests/spec/test_llm.py::test_llm_module_import_does_not_touch_anthropic_or_network \
	--deselect tests/spec/test_llm.py::test_rules_prompt_is_static

ASAN_PY_SKIP := --ignore=tests/spec/test_nonfunctional.py \
	--deselect tests/spec/test_determinism.py::test_replay_identical_across_processes \
	--deselect tests/spec/test_audit_v021.py::test_restore_rejects_corrupt_snapshots_without_crashing \
	--deselect tests/spec/test_audit_v021.py::test_spawn_extreme_coordinates_clamped \
	--deselect tests/spec/test_audit_v021.py::test_stress_skeleton_armies_throughput \
	--deselect tests/spec/test_obs_v3.py::test_v3_stress_throughput \
	--deselect tests/spec/test_obs_v3.py::test_v3_engine_ticks_per_second_random_decks \
	--deselect tests/spec/test_env_robust.py::test_multiprocessing_backend_smoke \
	--deselect tests/builder/test_determinism_soak.py::test_hash_is_stable_across_processes \
	--deselect tests/builder/test_symmetry_and_source.py::test_codegen_is_deterministic_and_committed \
	--deselect tests/builder/test_env_bots_policy.py::test_config_loads_and_train_cli \
	--deselect tests/builder/test_league_phase_d.py::test_league_train_run_history_and_eval_compatibility \
	--deselect tests/spec/test_league.py::test_league_train_snapshots_and_state \
	--deselect tests/spec/test_league.py::test_league_train_resume_total_is_absolute \
	--deselect tests/spec/test_league.py::test_best_response_probe \
	--deselect tests/spec/test_metagame.py::test_tournament_script \
	$(ASAN_PY_SKIP_SUBPROC)


asan-py: debug
	PUFFERROYALE_BINDING_DIR=build/debug/pufferroyale $(PRELOAD)=$(ASAN_RT) $(ASAN_ENV) \
		$(PYTHON) -m pytest -q -p no:cacheprovider tests/builder tests/spec $(ASAN_PY_SKIP)

test: test-c build pytest

clean:
	rm -rf build pufferroyale/*.so
