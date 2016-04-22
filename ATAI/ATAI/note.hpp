//
//  note.hpp
//  ATAI
//
//  Created by Nassim Versbraegen on 22/04/16.
//  Copyright © 2016 Nassim Versbraegen. All rights reserved.
//

#ifndef note_hpp
#define note_hpp

#include <stdio.h>
#include <iostream>


class note {
public:
    int rhythm;
    int noteheight;
    
    
    
    note(int Rhythm, int Noteheight){
        rhythm=Rhythm;
        noteheight=Noteheight;
    }
    
    friend std::ostream &operator<<( std::ostream &output,
                               const note *n )
    {
        output << "rhythm: " << n->rhythm << "\t note: " << n->noteheight;
        return output;
    }
    
    
};

#endif /* note_hpp */
