#install.packages("mailR")
library(mailR)


## REMEMBER!! TURN ON the Access for less secure apps https://www.google.com/settings/security/lesssecureapps

getwd()
current_date = Sys.Date()



mail_subject = paste("Tesing Send Email as on current_date: ",current_date)
mail_body =  paste("Tesing Send Email as on current_date: ",current_date)
mail_attach = "D:/Kaushik/Uconn Related/Study and Research/My Projects/SendEmail/test_attach.txt"
mail_host = "smtp.gmail.com"
mail_port = 465
mail_recipients = c("kaushikvijayakumar@gmail.com")
mail_sender = "kaushikvijayakumar@gmail.com"
mail_password = "Tolby@2009"
x = 10

fn_send_mail = function(x) {
  library(mailR)
  
  send.mail(from = mail_sender,
            to = mail_recipients,
            subject =  mail_subject,
            body =  mail_body,
            smtp = list(host.name = mail_host, port = mail_port, 
                        user.name = mail_sender,            
                        passwd = mail_password, ssl = TRUE),
            attach.files = mail_attach,
            authenticate = TRUE,
            send = TRUE)
  return(x)
}

fn_send_mail(10)
