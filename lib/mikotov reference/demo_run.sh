
###############################################################################################
# MODIFIED:
# --------
# Script for training good word and phrase vector model using public corpora, version 1.0.
# The training time will be from several hours to about a day.
#
# ~~Downloads about 8 billion words, makes phrases using two runs of word2phrase, trains
# a 500-dimensional vector model and evaluates it on word and phrase analogy tasks.~~
#
###############################################################################################

# This function will convert text to lowercase and remove special characters
normalize_text() {
  awk '{print tolower($0);}' | sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
  -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
  -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
  -e 's/«/ /g' | tr 0-9 " "
}

mkdir tmp
normalize_text < ../../data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00053-of-00100 >> ./tmp/data.txt

#gcc word2vec_mikotov.c -o ./tmp/word2vec -lm -pthread -O3 -march=native -funroll-loops
#gcc word2phrase_mikotov.c -o ./tmp/word2phrase -lm -pthread -O3 -march=native -funroll-loops

#./tmp/word2phrase -train ./tmp/data.txt -output ./tmp/data-phrase.txt -threshold 200 -debug 2
#./tmp/word2phrase -train ./tmp/data-phrase.txt -output ./tmp/data-phrase2.txt -threshold 100 -debug 2

./tmp/word2vec -train ./tmp/data-phrase2.txt -output ./tmp/vectors.bin -cbow 1 -size 500 -window 10 -negative 10 -hs 0 -sample 1e-5 -threads 40 -binary 1 -iter 3 -min-count 10 -dry-run 1
