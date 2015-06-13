.PHONY: docs
docs:
	rustdoc src/lib.rs --crate-name=imagefmt\
		--cfg feature=\"png\"\
		--cfg feature=\"tga\"\
		--cfg feature=\"bmp\"\
		--cfg feature=\"jpeg\"\
		-o docs -L target/debug -L target/debug/deps
