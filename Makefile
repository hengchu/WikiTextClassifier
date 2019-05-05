DIR := ${CURDIR}

default: image lambda

.PHONY: lambda
lambda:
	docker run --rm \
		-v ${DIR}/build:/io \
		-v ${DIR}/build.sh:/io/build.sh \
		-t cis700 bash /io/build.sh

.PHONY: image
image:
	docker build -t cis700 .

.PHONY: clean
clean:
	rm -rf build/*
