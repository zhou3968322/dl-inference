/**
* @program: dl_inference
*
* @description: ${description}
*
* @author: zhoubingcheng
*
* @email: bingchengzhou@foxmail.com
*
* @create: 2021-04-01 15:26
**/
//
// Created by 周炳诚 on 2021/4/1.
//

#include <stdexcept>
#include <iostream>

int compare( int a, int b ) {
    if ( a < 0 || b < 0 ) {
        throw std::invalid_argument( "received negative value" );
    }
}


int main(){
    try {
        compare( -1, 3 );
    }
    catch( const std::invalid_argument& e ) {
        std::count << "failed to compare" << std::endl;
        throw e;
    }
    return 1;
}