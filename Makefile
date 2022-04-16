docker-build:
	docker build -t hackaton-validate -f docker/Dockerfile.validate .

docker-run:
	docker run -it --rm --name hackaton-validate hackaton-validate
