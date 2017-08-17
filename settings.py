# @File(label="Select Root directory", style="directory") Root
import itertools
from ij import IJ
from ij.gui import GenericDialog
from java.awt import Panel
from java.util import Vector
import ConfigParser
import os
import sys
import ast

    
def settings():
    """ Settings """

    # Calls potential config file to set defaults
    # for settings dialog. Sets all defaults to 0
    # if no config file exists.
    con_path = os.path.join(str(Root), "FRET_params.cfg")
    if os.path.exists(con_path):
        dflt = config_read()
        print type(dflt.get("b_sub", "Rolling ball"))
        print (dflt.get("b_sub", "Rolling ball"))
        print type(int(dflt.get("b_sub", "Rolling ball")))
        print (dflt.get("b_sub", "Rolling ball"))
    else:
        dflt = {}

    feat_model_strings = ["Translation", "Rigid", 
                          "Similarity", "Affine
                          ]
    reg_model_strings = ["Translation", "Rigid",
                         "Similarity", "Affine",
                         "Elastic", "Least Squares"
                         ]
    b_sub_strings = ["Rolling Ball", "Manual Selection"]
                         
    # Registration parameters dialog.
    gd = GenericDialog("Advanced Settings")
    gd.addMessage("REGISTRATION PARAMETERS") 
    gd.addNumericField("Steps per scale octave: ", float(dflt.get("steps", 6)), 0, 7, "")
    gd.addNumericField("Max Octave Size: ", float(dflt.get("max_oct", 1024)), 0, 7, "")
    gd.addNumericField("Feature Descriptor Size: ", float(dflt.get("fd_size", 12)), 1, 7, "")
    gd.addNumericField("Initial Sigma: ", float(dflt.get("sigma", 1.2)), 2, 7, "")  
    gd.addNumericField("Max Epsilon: ", float(dflt.get("max_eps", 15)), 1, 7, "")
    gd.addNumericField("Min Inlier Ratio: ", float(dflt.get("min_inlier", 0.05)), 3, 7, "")
    gd.addCheckbox("Use Shrinkage Constraint", ast.literal_eval(dflt.get("shrinkage", "False")))
    gd.addChoice("Feature extraction model", feat_model_strings,
                 feat_model_strings[int(dflt.get("feat_model", 1))]
                 )
    gd.addChoice("Registration model", reg_model_strings,
                 reg_model_strings[int(dflt.get("reg_model", 1))]
                 )
    
    # Background removal parameters dialog.
    gd.addPanel(Panel())
    gd.addMessage("BACKGROUND REMOVAL") 
    gd.addChoice("Subtraction method:", b_sub_strings,
                 b_sub_strings[int(dflt.get("b_sub", 0))]
                 )          
    gd.addNumericField("Rolling ball size: ", 
                       float(dflt.get("ballsize", 50)), 1, 7, "px"
                       )
    gd.addCheckbox("Create Background", ast.literal_eval(dflt.get("create_b", "False")))
    gd.addCheckbox("Light Background", ast.literal_eval(dflt.get("light_b", "False")))
    gd.addCheckbox("Use Parabaloid", ast.literal_eval(dflt.get("parab", "False")))
    gd.addCheckbox("Do Pre-smoothing", ast.literal_eval(dflt.get("smooth", "False")))
    gd.addCheckbox("Correct Corners", ast.literal_eval(dflt.get("corners", "False")))

    # Measumrent parameters dialog.
    gd.addPanel(Panel())
    gd.addMessage("MEASUREMENT PARAMETERS")
    gd.addNumericField("Max Cell Area", float(dflt.get("cell_max", 2200)), 0, 7, "px")
    gd.addNumericField("Min Cell Area", float(dflt.get("cell_min", 200)), 0, 7, "px")
    gd.addNumericField("Ratio Subtraction", float(dflt.get("subtr_ratio", 0.31)), 3, 7, "")

    # Plot parameters dialog.
    gd.addPanel(Panel())
    gd.addMessage("PLOT PARAMETERS")
    gd.addNumericField("Max y, d and aFRET", float(dflt.get("p_max", 0.65)), 2, 7, "")
    gd.addNumericField("Min y, d and aFRET", float(dflt.get("p_min", 0.0)), 2, 7, "")
    gd.addNumericField("Max y, norm. d and aFRET", float(dflt.get("p_max_n", 1.65)), 2, 7, "")
    gd.addNumericField("Min y, norm. d and aFRET", float(dflt.get("p_min_n", 0.5)), 2, 7, "")
    
    # Set location of dialog on screen.
    #gd.setLocation(0,1000)
    
    gd.showDialog()
    
    # Checks if cancel was pressed, kills script.
    if gd.wasCanceled() is True:
        sys.exit("Cancel was pressed, script terminated.")

    # Parameters dictionary.
    parameters = {"steps" : gd.getNextNumber(), 
                 "max_oct" : gd.getNextNumber(),
                 "fd_size" : gd.getNextNumber(),
                 "sigma" : gd.getNextNumber(),
                 "max_eps" : gd.getNextNumber(),
                 "min_inlier" : gd.getNextNumber(),
                 "shrinkage" : gd.getNextBoolean(),
                 "feat_model" : gd.getNextChoiceIndex(),
                 "reg_model" : gd.getNextChoiceIndex(),
                 "b_sub" : gd.getNextChoiceIndex(),
                 "ballsize" : gd.getNextNumber(),
                 "create_b" : gd.getNextBoolean(),
                 "light_b" : gd.getNextBoolean(),
                 "parab" : gd.getNextBoolean(),
                 "smooth" : gd.getNextBoolean(),
                 "corners" : gd.getNextBoolean(),
                 "cell_max" : gd.getNextNumber(),
                 "cell_min" : gd.getNextNumber(),
                 "subtr_ratio" : gd.getNextNumber(),
                 "p_max" : gd.getNextNumber(),
                 "p_min" : gd.getNextNumber(),
                 "p_max_n" : gd.getNextNumber(),
                 "p_min_n" : gd.getNextNumber()
                 }

    parameters = config_write(parameters)   

    return parameters

def config_write(parameters):
    
    config = ConfigParser.RawConfigParser()
    config.add_section("Parameters")

    
    for key, value in parameters.iteritems():
        config.set("Parameters", str(key), str(value))

    with open(os.path.join(str(Root), "FRET_params.cfg"), "wb") as configfile:
        config.write(configfile)

    return parameters


def config_read():
    """ Config file reader, returns parameters from config file. """

    # Launch parser, set path to cfg file.
    config = ConfigParser.RawConfigParser()
    con_path = os.path.join(str(Root), "FRET_params.cfg")

    # Read config if .cfg exists.
    if os.path.exists(con_path):        
        config.read(con_path)

        # Read cfg section, return False if section ein't reel
        try:
            p_list = config.items("Parameters")
        except ConfigParser.NoSectionError:       
            print ("NoSectionError: "
                   "Section 'Parameters' not found, please "
                   "check (" + str(con_path) + ") for the "
                   "correct section name, or change the "
                   "section name in config_read()"
                   )
            raise

        # Build dict. of parameters.
        parameters = {}
        for key, value in p_list:
            parameters[key] = value

    else:
        print ("ERROR: Config file not found, check " + str(con_path) + " "
               "or select Advanced Settings to generate a new config file "
               )
        raise IOError("ERROR: Config file not found, check " + str(con_path) + " "
                      "or select Advanced Settings to generate a new config file "
                      )
    
    return parameters

if Adv_set == True:
    parameters = settings()
else: 
    parameters = config_read()