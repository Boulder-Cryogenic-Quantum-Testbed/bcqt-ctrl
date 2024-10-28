from urllib.request import urlopen
import sys

########################################
# Define a function to send an HTTP command and get the result
########################################

def Get_HTTP_Result(CmdToSend):

    # Specify the IP address of the switch box
    CmdToSend = "http://192.168.0.115/:" + CmdToSend

    # Send the HTTP command and try to read the result
    try:
        HTTP_Result = urlopen(CmdToSend, timeout=1)
        PTE_Return = HTTP_Result.read()

        # The switch displays a web GUI for unrecognised commands
        if len(PTE_Return) > 100:
            print ("Error, command not found:", CmdToSend)
            PTE_Return = "Invalid Command!"

    # Catch an exception if URL is incorrect (incorrect IP or disconnected)
    except:
        print ("Error, no response from device; check IP address and connections.")
        PTE_Return = "No Response!"
        sys.exit()      # Exit the script

    # Return the response
    return PTE_Return


########################################
# Send some commands to the switch box
########################################

print (Get_HTTP_Result("MN?"))        # Print model name
print (Get_HTTP_Result("SN?"))        # Print serial number

# SPDT switch models
status = Get_HTTP_Result("SETA=1")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position
status = Get_HTTP_Result("SETA=0")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position

status = Get_HTTP_Result("SETB=1")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position
status = Get_HTTP_Result("SETB=0")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position

status = Get_HTTP_Result("SETC=1")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position
status = Get_HTTP_Result("SETC=0")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position

status = Get_HTTP_Result("SETD=1")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position
status = Get_HTTP_Result("SETD=0")   # Set switch A
print ("Sw PORT:", Get_HTTP_Result("SWPORT?"))       # Print switch position

print ("Done.")
