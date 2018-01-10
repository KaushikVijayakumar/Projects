
library(taskscheduleR)
getwd()

setwd('D:/Kaushik/Uconn Related/Study and Research/My Projects/Automate R script/SendEmail.r')
setwd(dir = 'D:/Kaushik/Uconn Related/Study and Research/My Projects/Automate R script/SendEmail.r')

myscript <- system.file('extdata', 'C:/Users/kaush/OneDrive/Documents/SendEmail.r', package = 'taskscheduleR')
## run script once within 62 seconds
taskscheduler_create(taskname = 'myfancyscript', rscript = myscript, 
                     schedule = 'ONCE', starttime = format(Sys.time() + 62, '%H:%M'))
## run script every day at 09:10
taskscheduler_create(taskname = 'myfancyscriptdaily', rscript = myscript, 
                     schedule = 'DAILY', starttime = '09:10')

## delete the tasks
taskscheduler_delete(taskname = 'myfancyscript')
taskscheduler_delete(taskname = 'myfancyscriptdaily')

# When the task has run, you can look at the log which contains everything from stdout and stderr. The log file is located at the directory where the R script is located. 
## log file is at the place where the helloworld.R script was located

system.file('extdata', 'helloworld.log', package = 'taskscheduleR')
file.exists('SendEmail.r')

myscript$fil

customercare
