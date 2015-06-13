.PHONY: docs
docs:
	rustdoc src/lib.rs --crate-name=imagefmt\
		--cfg feature=\"png\"\
		--cfg feature=\"tga\"\
		--cfg feature=\"bmp\"\
		--cfg feature=\"jpeg\"\
		-o imagefmt-gh-pages -L target/debug -L target/debug/deps
