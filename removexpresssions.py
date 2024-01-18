# Python3 Program to implement
# the above approach
  
# Function to print strings present
# between any pair of delimiters
def printSubsInDelimiters(string) :
    
    # Stores the indices 
    newstring = "";
    flag1 = 0;
    flag2 = 0;
    itflag = 0;
    htmlflag = 0;
    i = 0;
    while i < len(string): 
        # If opening delimiter
        # is encountered
        if (string[i] == '$' and flag1 == 0 and flag2 == 0) :
            flag1 = 1;
        elif (string[i] == '$' and flag1 == 1 and flag2 == 0) :
            newstring = newstring + 'X';
            flag1 = 0;
        elif ((string[i] == '\\' or string[i] == '\n' or string[i] == '\r' or string[i] == '\t' or string[i] == '\b'or string[i] == '\f' or string[i] == '\a') and flag2 == 0 and flag1 == 0) :
            flag2 = 1;
            if (string[i] + string[i + 1] + string[i + 2] + string[i + 3] +string[i + 4] + string[i + 5] == '\textit') :
                itflag = 1;
                i = i + 6;
            elif (string[i] + string[i + 1] + string[i + 2] + string[i + 3] +string[i + 4]  == '\html') :
                htmlflag = 1;
                i = i + 18;                
                flag2=1;

        elif (string[i] == '}' and flag2 == 1 and flag1 == 0) :
            if(itflag == 1) :
                itflag = 0;
                flag2 = 0;
            elif(htmlflag == 1) :
                htmlflag = 2;
                flag2 = 1;
                flag1 =0;
            elif(htmlflag == 2) :
                htmlflag = 0;
                flag2 = 0
                flag1 =0;
                newstring = newstring + 'X';
                flag2 = 0;
            else:
                newstring = newstring + 'X';
                flag2 = 0;
        elif ((flag2 == 0 and flag1 == 0) or itflag == 1) :
            newstring = newstring + string[i];
        i = i + 1;
    print(newstring);



    
# Driver Code
if __name__ == "__main__" :
    EXIT_COMMAND = "exit"
    while (True):
        string = input('String\n') 
        if (string == EXIT_COMMAND):
            print("Exiting serial terminal.")
            break
        print('\n')
        printSubsInDelimiters(string);
        print('\n')

            
            # Insert your code here to do whatever you want with the input_str.

        # The rest of your program goes here.

    print("End.")
    
    #string = "Multiple-input and multiple-output (\htmladdnormallink{MIMO}{https://en.wikipedia.org/wiki/MIMO}) antenna technology is used to exploit the \htmladdnormallink{multi-path propagation}{\ray} to improve the capacity of a communication link. In principle, the link's capacity can be increased merely by increasing the power of the transmitting antenna. However – apart from being energy-consuming – this also increases interference to any other receivers should they transmit in the same frequency band. In MIMO, the energy is divided among multiple antennas, and the link capacity is improved without using any extra energy and without increasing interference towards other transmitters." 
    
    # This code is contributed by AnkThon
