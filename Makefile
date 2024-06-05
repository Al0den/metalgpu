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
ifdef msg
	@echo "---Pushing to git with commit message: '$(msg)'---"
	git add .
	git commit -m "$(msg)"
	git push
	@echo "---Pushed to git.---"
	make push_pypi
else
	@echo "---No commit message provided., not pushing ---"
endif

all:
	make recompile_c
	make testing
	make git_push
