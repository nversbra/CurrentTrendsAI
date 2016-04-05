source("plot_conf.R") #script for plotting dissimilarity matrix
#notesAsChars <- lapply(notes, intToUtf8)

# read a csv file and return a vector of the notes or of the durations
## type : 'duration' or 'notes'
inputSong <- function(csvFile, type){
  song_with_header <- read.csv(csvFile, header=TRUE, row.names=NULL)
  song <- song_with_header[7:nrow(song_with_header)-2, ]
  even_rows <- song[c(TRUE,FALSE), ]
  
  if (type == "notes"){
    notes <- as.numeric(paste(even_rows[,5])) #column 5 contains the notes 
    return (notes)
  }
  
  if (type == "durations"){
    moments <- as.numeric(paste(even_rows[,2])) #column 2 contains the moments the notes are played
    if(length(moments) %% 2 == 0) #if the length of the moments vector is even, it contains length/2 notes
      durations <- array(rep(0, length(moments)/2))
    else{ #if the length of the moment vectors is odd, it contains floor(length/2) + 1 notes 
      durations <- array(rep(0, floor(length(moments)/2) + 1))
    } 
      
    index_durations <- 1
    start <- 0
    for(i in 2:length(moments)){
      if (i%%2 == 1)
        start <- moments[i]
      else{
        durations[index_durations] <- (moments[i] - start) / 48 #
        index_durations <- index_durations + 1
      }
    }
    return(durations)  
  }
}



# create a list with the ngrams of a song 
computeNgrams <- function(n, data) {
  ngrams <- c() 
  for(index in 1:(length(data) - n + 1)) {
    ngrams <- c(ngrams, list(data[index : (index+n-1)]))
  }
  return(ngrams)
}

# create a table with the normalised counts for each ngram of a song
## the table has dimensions of the (value range)^n (lowest occuring note/duration to highest occuring note/duration in the dataset) to reduce the dimensionality 
## (otherwise it would e.g. have dimenions of (127)^n, with 127 the full MIDI note range)
countNgrams <- function(ngrams, valRange, minVal){
  counts <- array(rep(0, valRange*valRange*valRange), dim=c(valRange, valRange, valRange))
  for(index in 1:length(ngrams)) {
    #map the notes to the noterange or durationrange (e.g. when the note is 65 and the note range is 45-95, then the note maps to index 20)
    counts[ngrams[[index]][1] - minVal, ngrams[[index]][2] - minVal, ngrams[[index]][3] - minVal ] <- counts[ngrams[[index]][1] - minVal, ngrams[[index]][2] - minVal, ngrams[[index]][3] - minVal] + 1/length(ngrams)
  }
  return(counts)
}


dissimilarity <- function(song1, song2, noteRange, minNote){
  ngrams1 <- computeNgrams(3,song1)
  ngrams2 <- computeNgrams(3,song2)
  counts1 <- countNgrams(ngrams1, noteRange, minNote)
  counts2 <- countNgrams(ngrams2, noteRange, minNote)
  dissim <- 0
  for(i in 1:noteRange){
    for(j in 1:noteRange){
      for(k in 1:noteRange){
        dissim <- dissim + (counts1[i,j,k]-counts2[i,j,k])^2
      }
    }
  }
  return(dissim)
}


#read in all songs
numberOfSongs <- 40
songs <- list()
maxVal <- 0
#minVal <- 127 #largest midi note number
minVal <- 1000 #presumably larger than the longest note duration

for (s in 1:numberOfSongs){
  print(s)
  songs[[s]] <-  inputSong(paste0("songs-csv/",toString(s),".csv"), "durations")
  currentMax <- max(songs[[s]])
  currentMin <- min(songs[[s]])
  if (currentMax > maxVal){
    maxVal <- currentMax
  }
  if (currentMin < minVal){
    minVal <- currentMin
  }
    
}
# 
valRange <- maxVal - minVal 

dissimMatrix <- array(rep(NaN, numberOfSongs*numberOfSongs), dim=c(numberOfSongs, numberOfSongs))

for (i in 1:numberOfSongs){
  for (j in 1:numberOfSongs){
    if (i <= j)
      dissimMatrix[i,j] <- dissimilarity(songs[[i]], songs[[j]], valRange, minVal)
    else 
      dissimMatrix[i,j] <- dissimMatrix[j,i]
  }
}

print(valRange)
#print(dissimMatrix)
myImagePlot(dissimMatrix)















