#
library(rcdk)
library(tidyverse)
library(magrittr)
library(caret)
library(shiny)

load('RB.RData')

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
   
  # Biodegradability half-life / BioHL
  
  # Boiling point / BP
  
  # Readily biodegradable / RB
  observeEvent(input$execute, {
    
    output$smiles_string <- renderText(isolate({input$smiles2model}))
    
    output$ready_biodegradable <- renderText({
      
      mols <- parse.smiles(isolate({input$smiles2model}))[[1]]
      
      M <- eval.desc(mols, descNames)
      M <- predict(preProcValues, M)
      
      as.character(predict(RBioDeg_rf, M))
      
    })

    
  })

  
  
  
  
})
