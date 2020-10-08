#

library(shiny)

# UI
shinyUI(fluidPage(
  titlePanel("Ecotoxicological Profiling"),
  
  verticalLayout(
    splitLayout(wellPanel(
      textInput(
        inputId = 'smiles2model',
        label = 'Enter SMILES string representation of molecule:',
        value = ''
      )
    ),
    wellPanel(
      checkboxGroupInput(
        "models",
        "Select model(s):",
        c(
          # "Biodegradability half-life" = "BioHL",
          # "Boiling point" = "BP",
          "Readily biodegradable" = "RB"
        ),
        selected = ""
      )
    )),
    wellPanel(actionButton(inputId = 'execute',
                           label = 'Execute')),
    textOutput(outputId = 'smiles_string'),
    tags$h5('NRB = not readily biodegradable; RB = readily biodegradable'),
    textOutput(outputId = "ready_biodegradable")
  )
))