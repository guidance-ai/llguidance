TARGET = ../target/release

all:
	cd ../c && cargo build --release
	# force static linking
	rm -f $(TARGET)/libllguidance.so $(TARGET)/libllguidance.dylib $(TARGET)/libllguidance.dll
	c++ -W -Wall -std=c++20 -o $(TARGET)/c_sample c_sample.cpp -I../c -L$(TARGET) -lllguidance
	$(TARGET)/c_sample ../sample_parser/data/blog.schema.ll.json ../sample_parser/data/blog.sample.json
