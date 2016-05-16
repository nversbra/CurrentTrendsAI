//
//  song.cpp
//  ATAI
//
//  Created by Nassim Versbraegen on 05/04/16.
//  Copyright © 2016 Nassim Versbraegen. All rights reserved.
//

#include "song.hpp"


int song::round_to_rhythm(int x){
    switch (x) {
        case 0:
            x=1;    //this shouldnt happen unless smaller units than 1/128 appear
            break;
        case 1:     //      1/128
            break;
        case 2:     //      1/64
            break;
        case 4:     //      1/32
            break;
        case 8:     //      1/16
            break;
        case 16:    //      1/8
            break;
        case 32:    //      1/4
            break;
        case 64:    //      1/2
            break;
        case 128:   //      1/1
            break;
        case 3:     //      1/64.
            break;
        case 6:     //      1/32.
            break;
        case 12:    //      1/16.
            break;
        case 24:    //      1/8.
            break;
        case 48:    //      1/4.
            break;
        case 96:    //      1/2.
            break;
        case 192:   //      1/1.
            break;
        default:
            if (x>192){    //everything bigger than 1/1. gets mapped to 193
                x=193;
            } else {
                int mi[]= {1,2,3,4,6,8,12,16,24,32,48,64,96,128,192};
                std::vector<int> sortvec (mi, mi + sizeof(mi) / sizeof(int) );
                sortvec.push_back(x);
                std::sort (sortvec.begin(), sortvec.end());
                //find x in vec, compare to its neighbours, see which is closer, restart the whole switch with the value of neighbour
                for (int j=0; j<sortvec.size();j++) {
                    if (sortvec[j]==x){
                        int l = sortvec[j-1];
                        int r = sortvec[j+1];
                        int dl = x-l;
                        int dr = r -x;
                        if (dl<dr) {
                            x=l; //restart
                            break;
                        } else {
                            x=r; //restart
                            break;
                        }
                        
                    }
                }
            }
            
            break;
    }
    return x;
}


void song::read_targets(){
    std::ifstream file(target_file);
    std::string value;
    int ctr=0;
    while ( file.good() ){
        if (ctr++>0) {
            std::getline( file, value);
            if (value!= "") {
                std::string a=value.substr(0,value.find(";"));
                std::string b=value.substr(value.find(";")+1, value.size());
                std::string artist=b.substr(0, b.find(";"));
                int t = stoi(a);
                target_files.push_back(t);
                artists.push_back(artist);
            }
        } else{
            std::getline( file, value);
        }
    }
}


void song::buil_durations(){
    for (auto i:target_files) {
        std::vector<int> moments;
        std::vector<int> notes;
        std::string song=fileloc;
        song.append("/");
        song.append(std::to_string(i));
        song=song.append(".csv");
        //std::cout<<song<<"\n";
        std::ifstream file(song);
        std::string value;
        std::string time;
        std::string noteh;
        int ctr=0;
        while ( file.good() ){
            if (ctr++>4) {
                std::getline( file, value);
                if (ULONG_MAX != value.find("Time_signature"))
                {} else {
                    std::string a=value.substr(0,value.size());
                    int n=stoi(a);
                    if (n!=0) {
                        value=value.substr(value.find(",")+2,value.size());
                        if (ULONG_MAX != value.find("End_track"))
                        {break;}
                        if (ctr % 2== 0) {
                            noteh= value.substr(value.find(",")+2, value.find_last_of(","));
                            noteh=noteh.substr(noteh.find(",")+2,noteh.size());
                            noteh=noteh.substr(noteh.find(",")+2,noteh.size());
                            noteh=noteh.substr(0,noteh.find(","));
                            //noteh= value.substr(value.find_last_of(",")+2, value.size());
                            int n=stoi(noteh);
                            notes.push_back(n);
                        }
                        time = value.substr(0,value.find(","));
                        int t = stoi(time);
                        moments.push_back(t);
                    } else {break;}}
            } else {
                std::getline( file, value);
            }
        }
        std::vector<note*> v;
        for (int j=0; j<moments.size()-1; j+=2) {
            float x = moments[j+1];
            float y = moments[j];
            float f= (x-y)/6;
            int r= round(f);
            note *n= new note(r, notes[j/2]);
            v.push_back(n);
        }
        durations.push_back(v);
    }
}


void song::clean_durations(){
    for (int i=0; i<durations.size(); i++) {
        for (int j=0; j<durations[i].size(); j++) {
            durations[i][j]->rhythm=round_to_rhythm(durations[i][j]->rhythm);
        }
    }
}


void song::print_durations(){
    for (int i=0; i<durations.size(); i++) {
        std::cout<<"\n**------------song:" << i+1<< "-------------**\n";
        for (int j=0; j<durations[i].size(); j++) {
            std::cout<<"\n" << durations[i][j];
        }
    }
    
}