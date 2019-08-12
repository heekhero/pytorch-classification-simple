URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip
ZIP_FILE=./data/apple2orange.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data
rm $ZIP_FILE
mv ./data/apple2orange/trainA ./data/train/Apple
mv ./data/apple2orange/trainB ./data/train/Orange
mv ./data/apple2orange/testA ./data/test/Apple
mv ./data/apple2orange/testB ./data/test/Orange
rm -rf ./data/apple2orange

