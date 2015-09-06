.PHONY: docs

FEATURES =\
	--cfg feature=\"png\"\
	--cfg feature=\"tga\"\
	--cfg feature=\"bmp\"\
	--cfg feature=\"jpeg\"
DIRS = -L target/debug -L target/debug/deps

doc: doctest
	rustdoc src/lib.rs $(FEATURES) $(DIRS) --crate-name=imagefmt

doctest:
	rustdoc src/lib.rs $(FEATURES) $(DIRS) --test
