##########################################################################################
#                                Start Your Analyses                                     #
##########################################################################################
# **Load the R package for text mining and then load your texts into R.**
library(tm)   
docs <- Corpus(DirSource("C:\\Users\\Chirantan\\Desktop\\dataset\\final"))   
## Preprocessing      
docs <- tm_map(docs, removePunctuation)   # *Removing punctuation:*    
docs <- tm_map(docs, removeNumbers)      # *Removing numbers:*    
docs <- tm_map(docs, tolower)   # *Converting to lowercase:*    
docs <- tm_map(docs, removeWords, stopwords("english"))   # *Removing "stopwords" 
library(SnowballC)   
docs <- tm_map(docs, stemDocument)   # *Removing common word endings* (e.g., "ing", "es")   
docs <- tm_map(docs, stripWhitespace)   # *Stripping whitespace   
docs <- tm_map(docs, PlainTextDocument)   
## *This is the end of the preprocessing stage.*   


### Stage the Data      
dtm <- DocumentTermMatrix(docs)   
tdm <- TermDocumentMatrix(docs)   

### Explore your data      
freq <- colSums(as.matrix(dtm))   
length(freq)   
ord <- order(freq)   
m <- as.matrix(dtm)   
dim(m)   
write.csv(m, file="DocumentTermMatrix.csv")   
### FOCUS - on just the interesting stuff...   
#  Start by removing sparse terms:   
dtms <- removeSparseTerms(dtm, 0.1) # This makes a matrix that is 10% empty space, maximum.   
### Word Frequency   
head(table(freq), 20)   
# The above output is two rows of numbers. The top number is the frequency with which 
# words appear and the bottom number reflects how many words appear that frequently. 
#
tail(table(freq), 20)   
# Considering only the 20 greatest frequencies
#
# **View a table of the terms after removing sparse terms, as above.
freq <- colSums(as.matrix(dtms))   
freq   
# The above matrix was created using a data transformation we made earlier. 
# **An alternate view of term frequency:**   
# This will identify all terms that appear frequently (in this case, 50 or more times).   
findFreqTerms(dtm, lowfreq=50)   # Change "50" to whatever is most appropriate for your data.
#
#
#   
### Plot Word Frequencies
# **Plot words that appear at least 50 times.**   
library(ggplot2)   
wf <- data.frame(word=names(freq), freq=freq)   
p <- ggplot(subset(wf, freq>300), aes(word, freq))    
p <- p + geom_bar(stat="identity")   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))   
p   
#  
## Relationships Between Terms
### Term Correlations
# See the description above for more guidance with correlations.
# If words always appear together, then correlation=1.0.    
findAssocs(dtm, c("comedy"), corlimit=1) # specifying a correlation limit of 0.98   
findAssocs(dtm, c("horror"), corlimit=1)
findAssocs(dtm, c("romantic"), corlimit=1)
findAssocs(dtm, c("action"), corlimit=1)
findAssocs(dtm, c("movie"), corlimit=1)


#

# Change "question" & "analysi" to terms that actually appear in your texts.
# Also adjust the `corlimit= ` to any value you feel is necessary.
#
# 
### Word Clouds!   
# First load the package that makes word clouds in R.    
library(wordcloud)   
dtms <- removeSparseTerms(dtm, 0.15) # Prepare the data (max 15% empty space)   
freq <- colSums(as.matrix(dtm)) # Find word frequencies   
dark2 <- brewer.pal(6, "Dark2")   
wordcloud(names(freq), freq, max.words=100, rot.per=0.2, colors=dark2)    

### Clustering by Term Similarity

### Hierarchal Clustering   
dtms <- removeSparseTerms(dtm, 0.15) # This makes a matrix that is only 15% empty space.
library(cluster)   
d <- dist(t(dtms), method="euclidian")   # First calculate distance between words
fit <- hclust(d=d, method="ward")   
plot.new()
plot(fit, hang=-1)
groups <- cutree(fit, k=5)   # "k=" defines the number of clusters you are using   
rect.hclust(fit, k=5, border="red") # draw dendogram with red borders around the 5 clusters   

### K-means clustering   
library(fpc)   
library(cluster)  
dtms <- removeSparseTerms(dtm, 0.15) # Prepare the data (max 15% empty space)   
d <- dist(t(dtms), method="euclidian")   
kfit <- kmeans(d, 2)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)  

#########################################################################################

#import libraries to work with
library(plyr)
library(stringr)
library(e1071)

#load up word polarity list and format it
afinn_list <- read.delim(file='C:\\Users\\Chirantan\\Desktop\\data\\sentiment_analysis-master\\AFINN\\AFINN-111.txt', header=FALSE, stringsAsFactors=FALSE)
names(afinn_list) <- c('word', 'score')
afinn_list$word <- tolower(afinn_list$word)

#categorize words as very negative to very positive and add some movie-specific words
vNegTerms <- afinn_list$word[afinn_list$score==-5 | afinn_list$score==-4]
negTerms <- c(afinn_list$word[afinn_list$score==-3 | afinn_list$score==-2 | afinn_list$score==-1], "second-rate", "moronic", "third-rate", "flawed", "juvenile", "boring", "distasteful", "ordinary", "disgusting", "senseless", "static", "brutal", "confused", "disappointing", "bloody", "silly", "tired", "predictable", "stupid", "uninteresting", "trite", "uneven", "outdated", "dreadful", "bland")
posTerms <- c(afinn_list$word[afinn_list$score==3 | afinn_list$score==2 | afinn_list$score==1], "first-rate", "insightful", "clever", "charming", "comical", "charismatic", "enjoyable", "absorbing", "sensitive", "intriguing", "powerful", "pleasant", "surprising", "thought-provoking", "imaginative", "unpretentious")
vPosTerms <- c(afinn_list$word[afinn_list$score==5 | afinn_list$score==4], "uproarious", "riveting", "fascinating", "dazzling", "legendary")

#load up positive and negative sentences and format
posText <- read.delim(file='C:\\Users\\Chirantan\\Desktop\\data\\sentiment_analysis-master\\polarityData\\rt-polaritydata\\rt-polarity-pos.txt', header=FALSE, stringsAsFactors=FALSE)
posText <- posText$V1
posText <- unlist(lapply(posText, function(x) { str_split(x, "\n") }))
negText <- read.delim(file='C:\\Users\\Chirantan\\Desktop\\data\\sentiment_analysis-master\\polarityData\\rt-polaritydata\\rt-polarity-neg.txt', header=FALSE, stringsAsFactors=FALSE)
negText <- negText$V1
negText <- unlist(lapply(negText, function(x) { str_split(x, "\n") }))

#function to calculate number of words in each category within a sentence
sentimentScore <- function(sentences, vNegTerms, negTerms, posTerms, vPosTerms){
  final_scores <- matrix('', 0, 5)
  scores <- laply(sentences, function(sentence, vNegTerms, negTerms, posTerms, vPosTerms){
    initial_sentence <- sentence
    #remove unnecessary characters and split up by word 
    sentence <- gsub('[[:punct:]]', '', sentence)
    sentence <- gsub('[[:cntrl:]]', '', sentence)
    sentence <- gsub('\\d+', '', sentence)
    sentence <- tolower(sentence)
    wordList <- str_split(sentence, '\\s+')
    words <- unlist(wordList)
    #build vector with matches between sentence and each category
    vPosMatches <- match(words, vPosTerms)
    posMatches <- match(words, posTerms)
    vNegMatches <- match(words, vNegTerms)
    negMatches <- match(words, negTerms)
    #sum up number of words in each category
    vPosMatches <- sum(!is.na(vPosMatches))
    posMatches <- sum(!is.na(posMatches))
    vNegMatches <- sum(!is.na(vNegMatches))
    negMatches <- sum(!is.na(negMatches))
    score <- c(vNegMatches, negMatches, posMatches, vPosMatches)
    #add row to scores table
    newrow <- c(initial_sentence, score)
    final_scores <- rbind(final_scores, newrow)
    return(final_scores)
  }, vNegTerms, negTerms, posTerms, vPosTerms)
  return(scores)
}

#build tables of positive and negative sentences with scores
posResult <- as.data.frame(sentimentScore(posText, vNegTerms, negTerms, posTerms, vPosTerms))
negResult <- as.data.frame(sentimentScore(negText, vNegTerms, negTerms, posTerms, vPosTerms))
posResult <- cbind(posResult, 'positive')
colnames(posResult) <- c('sentence', 'vNeg', 'neg', 'pos', 'vPos', 'sentiment')
negResult <- cbind(negResult, 'negative')
colnames(negResult) <- c('sentence', 'vNeg', 'neg', 'pos', 'vPos', 'sentiment')

#combine the positive and negative tables
results <- rbind(posResult, negResult)

#run the naive bayes algorithm using all four categories
classifier <- naiveBayes(results[,2:5], results[,6])

#display the confusion table for the classification ran on the same data
confTable <- table(predict(classifier, results), results[,6], dnn=list('predicted','actual'))
confTable

#run a binomial test for confidence interval of results
binom.test(confTable[1,1] + confTable[2,2], nrow(results), p=0.5)

################################################################################################################

install.packages("devtools")
devtools::install_github("mjockers/syuzhet")
library(syuzhet)
library(pander)
my_example_text <- readLines("C:\\Users\\Chirantan\\Desktop\\dataset\\fold\\full-n-final.txt")
s_v <- get_sentences(my_example_text)
class(s_v)
## [1] "character"
str(s_v)
##  chr [1:12] "I begin this story with a neutral statement." ...
head(s_v)
sentiment_vector <- get_sentiment(s_v, method="bing")
sentiment_vector
afinn_vector <- get_sentiment(s_v, method="afinn")
afinn_vector
nrc_vector <- get_sentiment(s_v, method="nrc")
nrc_vector
sum(sentiment_vector)
mean(sentiment_vector)
summary(sentiment_vector)
nrc_data <- get_nrc_sentiment(s_v)
angry_items <- which(nrc_data$anger > 0)
s_v[angry_items]
joy_items <- which(nrc_data$joy > 0)
s_v[joy_items]
pander::pandoc.table(nrc_data[, 1:8])
valence <- (nrc_data[, 9]*-1) + nrc_data[, 10]
valence
barplot(
  sort(colSums(prop.table(nrc_data[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Sample text", xlab="Percentage"
)
