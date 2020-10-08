library(shiny)

shinyUI(fluidPage(# Application title
    titlePanel("EcoToxicological Profiling"),
    navlistPanel(
        tabPanel("Introduction", "qwerty01")
        , tabPanel("Data Input / Screen Selection", "qwerty02")
        , tabPanel("Atmospheric hydroxylation rate", "AOH")
        , tabPanel("Bioconcentration factor", "BCF")
        , tabPanel("Biodegradability half-life", "BioHL")
        , tabPanel("Boiling point", "BP")
        , tabPanel("Henry's Law Constant", "qwerty07")
        , tabPanel("Fish biotransformation half-life", "KM")
        , tabPanel("Octanol-air partition coefficient", "KOA")
        , tabPanel("Soil adsorption coefficient", "KOC")
        , tabPanel("Octanol-water partition coefficient", "logP")
        , tabPanel("Melting point", "MP")
        , tabPanel("Readily biodegradable", "RB")
        , tabPanel("Vapor pressure", "VP")
        , tabPanel("Water solubility", "WS")
        , tabPanel("data science / science des données", includeHTML("intro.html"))
    )))
