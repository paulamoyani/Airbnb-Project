library(dplyr)
library(data.table)


calendar= fread(file='C:/Users/User/Desktop/Term 2 MIBA/Data Driven Transformation/Challenge/Airbnb-Project/calendar_cleaned.csv', 
              header=TRUE,
              sep = ",",
              stringsAsFactors = FALSE,
              dec=".")

calendar[,available:=NULL]


Stats <- aggregate(calendar$price1 ~ calendar$listing_id, data = calendar, FUN = median)

setDT(Stats)

setnames(Stats,"calendar$listing_id","listing_id")
setnames(Stats,"calendar$price1","median_price")

complete_data <- merge.data.frame(Stats, calendar, 
                               by = 'listing_id',
                               all.x = TRUE)
setDT(complete_data)

complete_data[,special_date:=ifelse(price1>1.6*median_price,1,0),]

special_data=complete_data[special_date==1]

hack <- aggregate(special_data$special_date ~ special_data$date, data = calendar, FUN = sum)

setnames(hack,"special_data$special_date","special_date_sum")
setnames(hack,"special_data$date","date")
setDT(hack)

important_dates <- hack[special_date_sum>=60]

uniqueN(important_dates$date)
setDT(important_dates)
important_dates[,special_date_sum:=NULL,]
write.csv(important_dates,file = "Important_dates.csv")
