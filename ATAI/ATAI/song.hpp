//
//  song.hpp
//  ATAI
//
//  Created by Nassim Versbraegen on 05/04/16.
//  Copyright © 2016 Nassim Versbraegen. All rights reserved.
//

#ifndef rhythm_hpp
#define rhythm_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <cmath>
#include "note.hpp"

class song {
    std::string fileloc;
    std::vector<int> target_files;
    std::string target_file;
    
   
public:
    std::vector<std::vector<note*>> durations;
    std::vector<std::string> artists;
    int round_to_rhythm(int x);
    void clean_durations();
    void buil_durations();
    void print_durations();
    void read_targets();
    
    song(std::string file){
        std::string homedir = getenv("HOME");
        fileloc= homedir + file+ "songs-csv";
        target_file =homedir+ file+"dataset-balanced.csv";
        read_targets();
        buil_durations();
        clean_durations();
    }

   

};


#endif /* rhythm_hpp */
