import ROOT
from random import gauss, uniform

# 1. Create a ROOT file.
f = ROOT.TFile("nested_example.root", "RECREATE")

# Let's organize our data in two categories: A and B
categories = ['A', 'B']

for category in categories:
    # 2. Create a directory for each category.
    directory = f.mkdir(category)
    directory.cd()  # Change to the current directory
    
    # 3a. Create and fill a 1D histogram.
    h1 = ROOT.TH1F(f"h1_{category}", f"1D histogram for {category}", 100, -5, 5)
    for i in range(1000):
        h1.Fill(gauss(0, 1))  # Fill with Gaussian random numbers
    
    # 3b. Create and fill a 2D histogram.
    h2 = ROOT.TH2F(f"h2_{category}", f"2D histogram for {category}", 40, -5, 5, 40, -5, 5)
    for i in range(1000):
        h2.Fill(gauss(0, 1), gauss(0, 1))
    
    # 3c. Store some metadata using TNamed.
    meta1 = ROOT.TNamed("Author", "ChatGPT")
    meta2 = ROOT.TNamed("Description", f"Data for category {category}")
    meta3 = ROOT.TNamed("RandomValue", str(uniform(0, 1)))
    
    # Writing objects to the directory.
    h1.Write()
    h2.Write()
    meta1.Write()
    meta2.Write()
    meta3.Write()

# 4. Close the file.
f.Close()
