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
	@echo "---Pushing differences to github"
	git add .
	git commit -m "auto-push"
	git push
	@echo "---Pushed differences to github"

all:
	make recompile_c
	make push_pypi
	make git_push
	make testing
