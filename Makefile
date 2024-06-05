recompile_c:
	@echo "---Recompiling C library---"
	cd metal-gpu-c && $(MAKE) full
	@echo "---Recompiled C library---"
push_pypi:
	@echo "---Pushing to PyPi---"
	cd metalgpu && $(MAKE) push
	@echo "---Pushed to PyPi---"
testing:
	@echo "---Running tests---"
	cd examples && $(MAKE) testing
	@echo "---All tests passed---"
git_push:
	@echo "---Pushing to git with commit message: '$(COMMIT_MESSAGE)'---"
	git add .
	git commit -m "$(COMMIT_MESSAGE) -  auto"
	git push
	@echo "---Pushed to git.---"

all:
	make recompile_c
	make testing
	make git_push
	make push_pypi
