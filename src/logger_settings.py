'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package. 
@precondition:
@summary:
'''

import logging

def setLoggers(filename=None):
    logFormatter = logging.Formatter("%(asctime)s : %(levelname)s : %(module)s : %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    for h in rootLogger.handlers:
        rootLogger.removeHandler(h)

    if filename != None:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
