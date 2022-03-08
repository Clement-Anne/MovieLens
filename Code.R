#############General infos################
##Author: Clément ANNE
##Author e-mail: clement.anne90@gmail.com
##Date: March 08,2022
##########################################

################Outline###################
##  0. Preparation
##  1. Preprocessing
##    1.1. Splitting edx
##    1.2. Identifying genre types
##    1.3. Cleaning datasets
##    1.4. RMSE function
##  2. Training models on the edx set
##    2.1. Model 1: User and movie biases
##    2.2. Model 2: Adding time-varying movie biases
##    2.3. Model 3: Adding time-varying user biases
##    2.4. Model 4: Adding movie genres biases
##    2.5. Model 5: All biases with regularization
##    2.6. Summary
##  3. Applying models on the validation set
##    3.1. Model 1: User and movie biases
##    3.2. Model 2: Adding time-varying movie biases
##    3.3. Model 3: Adding time-varying user biases
##    3.4. Model 4: Adding movie genres biases
##    3.5. Model 5: All biases with regularization
##    3.6. Summary
##########################################


##############################
#
#       0. Preparation
#
##############################

###Installing packages when needed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")



###Loading packages
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(knitr)
library(tinytex)


###Install tinytex if needed
if(is_tinytex()==FALSE) tinytex::install_tinytex()

###Free memory in the environment
rm(list=ls())

###Indication of computation start time
computation_start <- now()

###Downloading the MovieLens dataset
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

###MovieLens dataset
movielens <- left_join(ratings, movies, by = "movieId")

### Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

### Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

### Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

###Remove temporary files
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##############################
#
#       1. Preprocessing
#
##############################

##############################
#      1.1. Splitting edx
##############################

###edx split (edx_test will be 10% of edx)
set.seed(1, sample.kind="Rounding") 
edx_test_id <- createDataPartition(edx$rating,times=1,p=0.1,list=FALSE)
edx_train <- edx[-edx_test_id]
temp <- edx[edx_test_id]

###Identification of rows in the test set without a movie ID or user ID in the train set
edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)
 
###Remove temporary files
rm(temp,removed,edx_test_id)

##############################
#  1.2. Identifying genre types
##############################

genres <- unique(edx$genres)
length(genres)

###No genres listed -> 1 unique film -> Consider is na (already film level bias)
genres[str_detect(genres,"\\(")==TRUE]
genres <- genres[str_detect(genres,"\\(")==FALSE] #Remove the NA genre

###Split genres in columns
genres_split <- str_split(genres,"\\|",simplify=TRUE)

###Identify unique genres
genres_types <- map_dfr(1:ncol(genres_split),function(i){
  tibble(genre=unique(genres_split[,i]))
})%>%
  unique()
genres_types <-genres_types[genres_types!=""]

rm(genres,genres_split)

##############################
#      1.3. Cleaning datasets
##############################

####Remove edx dataset
rm(edx)


###Cleaning the edx_train set
edx_train_clean <- edx_train%>%
  #Convert timestamp into date
  mutate(Date=as_datetime(timestamp)%>%as_date())%>%
  group_by(movieId)%>%
  #First date for a rating of movie i
  mutate(min_date_movie=min(Date))%>%
  ungroup()%>%
  #Date gap since 1st rating of movie i
  mutate(diff_date_movie=as.numeric(Date-min_date_movie))%>%
  group_by(userId)%>%
  #First date for a rating by user u
  mutate(min_date_user=min(Date))%>%
  ungroup()%>%
  #Date gap since 1st rating by user u
  mutate(diff_date_user=as.numeric(Date-min_date_user))%>%
  #Convert date gaps in full weeks units
  mutate(diff_date_user_w=floor(diff_date_user/7))%>%
  mutate(diff_date_movie_w=floor(diff_date_movie/7))

rm(edx_train)

###Cleaning the edx_test set
edx_test_clean <- edx_test%>%
  #Convert timestamp into date
  mutate(Date=as_datetime(timestamp)%>%as_date())%>%
  group_by(movieId)%>%
  #First date for a rating of movie i
  mutate(min_date_movie=min(Date))%>%
  ungroup()%>%
  #Date gap since 1st rating of movie i
  mutate(diff_date_movie=as.numeric(Date-min_date_movie))%>%
  group_by(userId)%>%
  #First date for a rating by user u
  mutate(min_date_user=min(Date))%>%
  ungroup()%>%
  #Date gap since 1st rating by user u
  mutate(diff_date_user=as.numeric(Date-min_date_user))%>%
  #Convert date gaps in full weeks units
  mutate(diff_date_user_w=floor(diff_date_user/7))%>%
  mutate(diff_date_movie_w=floor(diff_date_movie/7))

rm(edx_test)

###Cleaning the validation set
validation_clean <- validation%>%
  #Convert timestamp into date
  mutate(Date=as_datetime(timestamp)%>%as_date())%>%
  group_by(movieId)%>%
  #First date for a rating of movie i
  mutate(min_date_movie=min(Date))%>%
  ungroup()%>%
  #Date gap since 1st rating of movie i
  mutate(diff_date_movie=as.numeric(Date-min_date_movie))%>%
  group_by(userId)%>%
  #First date for a rating by user u
  mutate(min_date_user=min(Date))%>%
  ungroup()%>%
  #Date gap since 1st rating by user u
  mutate(diff_date_user=as.numeric(Date-min_date_user))%>%
  #Convert date gaps in full weeks units
  mutate(diff_date_user_w=floor(diff_date_user/7))%>%
  mutate(diff_date_movie_w=floor(diff_date_movie/7))

rm(validation)

##############################
#      1.4. RMSE function
##############################

##RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##############################
#      1.5. Data exploration
##############################

edx_clean <- bind_rows(edx_train_clean,edx_test_clean)

###Notes
# -This section has been built exclusively on the edx set
#   
###

###Distribution movie ratings
edx_clean%>%
  group_by(rating)%>%
  ggplot(aes(rating))+
  geom_bar(color="black",fill="brown")+
  scale_y_continuous(name="",breaks=seq(500000,2500000,500000),labels=c("500 000", "1 000 000", "1 500 000", "2 000 000", "2 500 000"))+
  scale_x_continuous(breaks=seq(0.5,5,0.5))+
  xlab("Movie rating")+
  ggtitle("Distribution of movie ratings")  

###Number of ratings per user
edx_clean%>%
  group_by(userId)%>%
  summarize(n=n())%>%
  select(userId,n)%>%
  distinct()%>%
  ungroup()%>%
  ggplot(aes(n))+
  geom_histogram(binwidth=.1,color="black",fill="brown")+
  scale_x_log10()+
  xlab("N of ratings per user")+
  ylab("")+
  ggtitle("Distribution of the number of ratings by user")

###Number of ratings per movie
edx_clean%>%
  group_by(movieId)%>%
  summarize(n=n())%>%
  select(movieId,n)%>%
  distinct()%>%
  ungroup()%>%
  ggplot(aes(n))+
  geom_histogram(binwidth=.1,color="black",fill="brown")+
  scale_x_log10()+
  xlab("N of ratings per movie")+
  ylab("")+
  ggtitle("Distribution of the number of ratings by movie")


###Relationship between ratings and time between rating and first movie rating regarding movie i by any user
edx_clean%>%
  group_by(diff_date_movie_w)%>%
  summarize(mean_rating=mean(rating))%>%
  ggplot(aes(diff_date_movie_w,mean_rating))+
  geom_point(alpha=.5,color="brown")+
  geom_smooth(method="loess")+
  xlab("N weeks since the first movie rating by any user")+
  ylab("")+
  ggtitle("Time-varying movie effect on average ratings")


###Relationship between ratings and time between rating and first movie rating by user i for any movie
edx_clean%>%
  group_by(diff_date_user_w)%>%
  summarize(mean_rating=mean(rating))%>%
  ggplot(aes(diff_date_user_w,mean_rating))+
  geom_point(alpha=.5,color="brown")+
  geom_smooth(method="loess")+
  xlab("N weeks since the first user rating for any movie")+
  ylab("")+
  ggtitle("Time-varying user effect on average ratings")

###Creation of genre logical vectors
genres_d_edx<- map_dfc(1:length(genres_types),function(i){
  str_detect(edx_clean$genres,genres_types[i])%>%
    as_tibble_col(column_name = genres_types[i])
})

###Product_rating function for mutate_at
#Goal: Replacing the logical vectors by the rating when movie i belongs to genre g
product_rating <- function(x){
  x*edx_clean$rating
}

###If_else function for mutate_at
#Goal: Replace the genre_rating vector by NA if movie i does not belong to genre g
if_else_zero <- function(x){
  ifelse(x==0,NA,x)
}

###Average ratings and movie genres
edx_clean%>%
  #Adding the genres logical vector
  bind_cols(genres_d_edx)%>%
  #Replacing by the rating in case movie i belongs to genre g
  mutate_at(.vars = genres_types,product_rating)%>%
  mutate_at(.vars = genres_types,if_else_zero)%>%
  #Mean rating by movie genre
  summarize_at(.vars = genres_types,mean,na.rm=TRUE)%>%
  #Tidy format
  pivot_longer(everything(),names_to = "Genre",values_to ="Mean_rating" )%>%
  #Reorder genres
  mutate(Genre=reorder(Genre,Mean_rating))%>%
  arrange(desc(Mean_rating))%>%
  #Graph
  ggplot(aes(Genre,Mean_rating))+
  geom_bar(stat="identity",col="black",fill="brown")+
  coord_flip()+
  ylab("Average rating")+
  xlab("")+
  ggtitle("Average rating by movie genre")


rm(edx_clean, genres_d_edx)

##############################
#
#       2. Training models on the edx set
#
##############################

##############################
#      2.1. Model 1: User and movie biases
##############################

###Notes
# -Training the model on edx_train 
#   and assessing him on edx_test for the time being
###

###Average rating
mu <- mean(edx_train_clean$rating)

###Movie bias
movie_avgs <- edx_train_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_train_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Predict rating
predicted_rating <- edx_test_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred=mu+b_i+b_u)%>%
  pull(pred)

###RMSE
rmse_model1_train <- RMSE(edx_test_clean$rating,predicted_rating)
rmse_results_train <- tibble(method = "Baseline (User and Movie effects)", 
                                 RMSE = rmse_model1_train)

##############################
#      2.2. Model 2: Adding time-varying movie biases 
##############################

###Notes
# -Training the model on edx_train 
#   and assessing him on edx_test for the time being
###

###Average rating
mu <- mean(edx_train_clean$rating)

###Movie bias
movie_avgs <- edx_train_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_train_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Time-varying movie bias
movie_time_avgs <- edx_train_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(diff_date_movie_w) %>%
  summarize(b_it = mean(rating - mu - b_i - b_u))

###Predict rating
predicted_rating <- edx_test_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w')%>%
  mutate(pred=mu+b_i+b_u+b_it)%>%
  pull(pred)

###RMSE
rmse_model2_train <- RMSE(edx_test_clean$rating,predicted_rating)
rmse_results_train <- bind_rows(rmse_results_train,
                                tibble(method = "Baseline + Movie-time effects", 
                                       RMSE = rmse_model2_train))

##############################
#      2.3. Model 3: Adding time-varying user biases
##############################

###Notes
# -Training the model on edx_train 
#   and assessing him on edx_test for the time being
###

###Average rating
mu <- mean(edx_train_clean$rating)

###Movie bias
movie_avgs <- edx_train_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_train_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Time-varying movie bias
movie_time_avgs <- edx_train_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(diff_date_movie_w) %>%
  summarize(b_it = mean(rating - mu - b_i - b_u))

###Time-varying user bias
user_time_avgs <- edx_train_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  group_by(diff_date_user_w) %>%
  summarize(b_ut = mean(rating - mu - b_i - b_u - b_it))  

###Predict rating
predicted_rating <- edx_test_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w')%>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  mutate(pred=mu+b_i+b_u+b_it+b_ut)%>%
  pull(pred)

###RMSE
rmse_model3_train <- RMSE(edx_test_clean$rating,predicted_rating)
rmse_results_train <- bind_rows(rmse_results_train,
                                tibble(method = "Baseline + (User-time and Movie-time) effects", 
                                       RMSE = rmse_model3_train))


##############################
#      2.4. Model 4: Adding movie genres biases
##############################

###Notes
# -Training the model on edx_train 
#   and assessing him on edx_test for the time being
###

###Average rating
mu <- mean(edx_train_clean$rating)

###Movie bias
movie_avgs <- edx_train_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_train_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Time-varying movie bias
movie_time_avgs <- edx_train_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(diff_date_movie_w) %>%
  summarize(b_it = mean(rating - mu - b_i - b_u))

###Time-varying user bias
user_time_avgs <- edx_train_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  group_by(diff_date_user_w) %>%
  summarize(b_ut = mean(rating - mu - b_i - b_u - b_it))  

###Movie genres bias
genres_avgs <- edx_train_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_it- b_ut))  

###Predict rating
predicted_rating <- edx_test_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w')%>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  left_join(genres_avgs,by='genres')%>%
  mutate(pred=mu+b_i+b_u+b_it+b_ut+b_g)%>%
  pull(pred)

###RMSE
rmse_model4_train <- RMSE(edx_test_clean$rating,predicted_rating)
rmse_results_train <- bind_rows(rmse_results_train,
                                tibble(method = "Baseline + (User-time, Movie-time, Genres) effects", 
                                       RMSE = rmse_model4_train))

###Remove temporary files
rm(movie_avgs,user_avgs,user_time_avgs,movie_time_avgs,
   genres_avgs,predicted_rating)

##############################
#      2.5. Model 5: All biases with regularization
##############################

###Vector of lambda values
lambdas <- seq(0,10,.25)

###RMSE vector depending on lambda
rmses <- sapply(lambdas,function(l){
  
  print(paste0("Lambda: ",l))   
  lambda <- l
  
  ###Average rating
  mu <- mean(edx_train_clean$rating)
  print("Average rating OK")
  
  ###Movie bias
  movie_avgs <- edx_train_clean %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  print("Movie bias OK")
  
  ###User bias
  user_avgs <- edx_train_clean %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))
  print("User bias OK")
  
  ###Time-varying movie bias
  movie_time_avgs <- edx_train_clean %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    group_by(diff_date_movie_w) %>%
    summarize(b_it = sum(rating - mu - b_i - b_u)/(n()+lambda))
  print("Movie time bias OK")
  
  ###Time-varying user bias
  user_time_avgs <- edx_train_clean%>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(movie_time_avgs, by='diff_date_movie_w') %>%
    group_by(diff_date_user_w) %>%
    summarize(b_ut = sum(rating - mu - b_i - b_u - b_it)/(n()+lambda))  
  print("User time bias OK")
  
  ###Movie genres bias
  genres_avgs <- edx_train_clean%>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(movie_time_avgs, by='diff_date_movie_w') %>%
    left_join(user_time_avgs,by='diff_date_user_w')%>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - mu - b_i - b_u - b_it- b_ut)/(n()+lambda))  
  print("Genre bias OK")
  
  ###Predict rating
  predicted_rating <- edx_test_clean%>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(movie_time_avgs, by='diff_date_movie_w')%>%
    left_join(user_time_avgs,by='diff_date_user_w')%>%
    left_join(genres_avgs,by='genres')%>%
    mutate(pred=mu+b_i+b_u+b_it+b_ut+b_g)%>%
    pull(pred)
  print("Prediction OK")
  
  return(RMSE(edx_test_clean$rating,predicted_rating))
})


###Plotting RMSE vs lambda
qplot(lambdas,rmses)+xlab(expression(lambda))+ylab("RMSE") 

###Optimal lambda parameter
lambda_star <- lambdas[which.min(rmses)]
print(paste0("Optimal lambda is ",as.character(lambda_star)))

rmse_model5_train <- min(rmses)
rmse_results_train <- bind_rows(rmse_results_train,
                                tibble(method = "Reg. Baseline + (User-time, Movie-time, Genres) effects", 
                                       RMSE = rmse_model5_train))

##############################
#      2.6. Summary
##############################

###Summing up results developed in the training set (edx)
kable(rmse_results_train)


##############################
#
#       3. Applying models on the validation set
#
##############################

###Merging the training set edx
edx_clean <- bind_rows(edx_train_clean,edx_test_clean)
rm(edx_train_clean,edx_test_clean)

##############################
#      3.1. Model 1: User and movie biases
##############################

###Notes
# -Training the model on edx 
#   and assessing him on validation
###


###Average rating
mu <- mean(edx_clean$rating)

###Movie bias
movie_avgs <- edx_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Predict rating
predicted_rating <- validation_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred=mu+b_i+b_u)%>%
  pull(pred)

###RMSE
rmse_model1_validation <- RMSE(validation_clean$rating,predicted_rating)
rmse_results_validation <- tibble(method = "Baseline (User and Movie effects)", 
                                 RMSE = rmse_model1_validation)

##############################
#      3.2. Model 2: Adding time-varying movie biases
##############################

###Notes
# -Training the model on edx_train 
#   and assessing him on edx_test for the time being
###

###Average rating
mu <- mean(edx_clean$rating)

###Movie bias
movie_avgs <- edx_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Time-varying movie bias
movie_time_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(diff_date_movie_w) %>%
  summarize(b_it = mean(rating - mu - b_i - b_u))

###Predict rating
predicted_rating <- validation_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w')%>%
  mutate(pred=mu+b_i+b_u+b_it)%>%
  pull(pred)

###RMSE
rmse_model2_validation <- RMSE(validation_clean$rating,predicted_rating)
rmse_results_validation <- bind_rows(rmse_results_validation,
                                     tibble(method = "Baseline + Movie-time effects", 
                                            RMSE = rmse_model2_validation))


##############################
#      3.3. Model 3: Adding time-varying user biases
##############################

###Notes
# -Training the model on edx 
#   and assessing him on validation
###

###Average rating
mu <- mean(edx_clean$rating)

###Movie bias
movie_avgs <- edx_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Time-varying movie bias
movie_time_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(diff_date_movie_w) %>%
  summarize(b_it = mean(rating - mu - b_i - b_u))

###Time-varying user bias
user_time_avgs <- edx_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  group_by(diff_date_user_w) %>%
  summarize(b_ut = mean(rating - mu - b_i - b_u - b_it))  

###Predict rating
predicted_rating <- validation_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w')%>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  mutate(pred=mu+b_i+b_u+b_it+b_ut)%>%
  pull(pred)

###RMSE
rmse_model3_validation <- RMSE(validation_clean$rating,predicted_rating)
rmse_results_validation <- bind_rows(rmse_results_validation,
                                     tibble(method = "Baseline +(User-time and Movie-time) effects", 
                                 RMSE = rmse_model3_validation))



##############################
#      3.4. Model 4: Adding movie genres biases
##############################

###Notes
# -Training the model on edx 
#   and assessing him on validation
###

###Average rating
mu <- mean(edx_clean$rating)

###Movie bias
movie_avgs <- edx_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

###User bias
user_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###Time-varying movie bias
movie_time_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(diff_date_movie_w) %>%
  summarize(b_it = mean(rating - mu - b_i - b_u))

###Time-varying user bias
user_time_avgs <- edx_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  group_by(diff_date_user_w) %>%
  summarize(b_ut = mean(rating - mu - b_i - b_u - b_it))  

###Movie genres bias
genres_avgs <- edx_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_it- b_ut))  

###Predict rating
predicted_rating <- validation_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w')%>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  left_join(genres_avgs,by='genres')%>%
  mutate(pred=mu+b_i+b_u+b_it+b_ut+b_g)%>%
  pull(pred)

###RMSE
rmse_model4_validation <- RMSE(validation_clean$rating,predicted_rating)
rmse_results_validation <- bind_rows(rmse_results_validation,
                                     tibble(method = "Baseline + (User-time, Movie-time, Genres) effects", 
                                            RMSE = rmse_model4_validation))

###Remove previous data
rm(genres_avgs,movie_avgs,movie_time_avgs,
   user_avgs,user_time_avgs,mu,predicted_rating)


##############################
#      3.5. Model 5: All biases with regularization
##############################


###Notes
# -Training the model on edx 
#   and assessing him on validation
#   Using lambda=5.25 found as optimal on the training set  
###


###Average rating
mu <- mean(edx_clean$rating)

###Movie bias
movie_avgs <- edx_clean %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda_star))

###User bias
user_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda_star))

###Time-varying movie bias
movie_time_avgs <- edx_clean %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(diff_date_movie_w) %>%
  summarize(b_it = sum(rating - mu - b_i - b_u)/(n()+lambda_star))

###Time-varying user bias
user_time_avgs <- edx_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  group_by(diff_date_user_w) %>%
  summarize(b_ut = sum(rating - mu - b_i - b_u - b_it)/(n()+lambda_star))  

###Movie genres bias
genres_avgs <- edx_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w') %>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_it- b_ut)/(n()+lambda_star))  

###Predict rating
predicted_rating <- validation_clean%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(movie_time_avgs, by='diff_date_movie_w')%>%
  left_join(user_time_avgs,by='diff_date_user_w')%>%
  left_join(genres_avgs,by='genres')%>%
  mutate(pred=mu+b_i+b_u+b_it+b_ut+b_g)%>%
  pull(pred)
print("Prediction OK")

###RMSE
rmse_model5_validation <- RMSE(validation_clean$rating,predicted_rating)
rmse_results_validation <- bind_rows(rmse_results_validation,
                                     tibble(method = "Reg. Baseline + (User-time, Movie-time, Genres) effects", 
                                            RMSE = rmse_model5_validation))
##############################
#      3.6. Summary
##############################

###Summing up results
kable(rmse_results_validation)

###Improvement from model 1 to model 5
(rmse_model1_validation-rmse_model5_validation)*100/rmse_model1_validation

###End of computation time
computation_end <- now()

###Computation time
computation_end-computation_start



