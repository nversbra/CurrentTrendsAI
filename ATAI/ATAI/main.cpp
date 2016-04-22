
//
//  main.cpp
//  ATAI
//
//  Created by Nassim Versbraegen on 05/04/16.
//  Copyright © 2016 Nassim Versbraegen. All rights reserved.
//

#include <iostream>
#include <string>
#include "song.hpp"


int main(int argc, const char * argv[]) {
    
    std::string files="/Documents/CurrentTrendsAI/";
    song *r = new song(files);                        //class containing the durations vector, which holds 'notes', which are simply pairs of rhythm and notes  
    r->print_durations();
    
    return 0;
}
