//
//  main.cpp
//  DeepBlueNote
//
//  Created by Nassim Versbraegen on 01/05/16.
//  Copyright © 2016 Nassim Versbraegen. All rights reserved.
//

#include <iostream>
#include <unistd.h>

int main(int argc, const char * argv[]) {
    char buffer[FILENAME_MAX];
    getcwd(buffer, FILENAME_MAX);
    
    std::string loc= std::string(argv[0]);
    std::string train= std::string(argv[1]);
    std::string test= std::string(argv[2]);
    std::string out= std::string(argv[3]);
    loc=loc.substr(0,loc.find("DeepBlueNote"));
    std::cout << buffer << "\n";
    std::cout << loc << "\n";
    std::string command = "cd "+loc+" \nPython DeepBlueNote.py "+train+" "+test+" "+out+" 97" +" 0.991127084577083" + " 0.6889224750972345" +" 7"+" 19"+" 0.05276771837066035"+" 0.005052682628063044" +" 29001";
    std::cout << command << "\n";
    std::system(command.c_str());
    return 0;
}
