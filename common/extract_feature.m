function feature = extract_feature(filepath)

s = audioread(filepath);

feature = s(1:100);