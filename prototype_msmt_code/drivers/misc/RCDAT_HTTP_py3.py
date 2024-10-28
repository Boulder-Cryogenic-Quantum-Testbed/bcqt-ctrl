from urllib.request import urlopen
import sys

########################################
# Define a function to send an HTTP command and get the result
########################################

def Get_HTTP_Result(CmdToSend):

    # Specify the IP address
    CmdToSend = "http://192.168.0.30/:" + CmdToSend

    # Send the HTTP command and try to read the result
    try:
        HTTP_Result = urlopen(CmdToSend, timeout=2)
        PTE_Return = HTTP_Result.read()

    # Catch an exception if URL is incorrect (incorrect IP or disconnected)
    except:
        print ("Error, no response from device; check IP address and connections.")
        PTE_Return = "No Response!"
        sys.exit()      # Exit the script

    # Return the response
    return PTE_Return


########################################
# Send commands / queries to the attenuator
########################################

print (Get_HTTP_Result("MN?"))        		# Print model name
print (Get_HTTP_Result("SN?"))        		# Print serial number

# Single channel models
Set_1 = Get_HTTP_Result("SETATT=10.75")   	# Set attenuation
print ("Set Attenuation:", str(Set_1))

# Multi-channel models
#Set_1 = Get_HTTP_Result("CHAN:1:SETATT:0")   	        # Set channel 1 to 0 dB
#Set_1 = Get_HTTP_Result("CHAN:2:3:4:SETATT:10.25")   	# Set channels 2, 3 and 4 to 10.25 dB
#print ("Set Attenuation:", str(Set_1))

Read_1 = Get_HTTP_Result("ATT?")        	# Read attenuation
print ("Read Attenuation: ", str(Read_1))
