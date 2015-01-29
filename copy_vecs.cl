#define ROW_STEP 4
#define COL_STEP 3

kernel void copy_vecs
(
    const int band_rows,
    const int band_cols,
    const int row_num,
    const int col_num,
    const float scale,
    const float top,
    global float *band,
    global float *vecs,
    global float *pos
)
{
    int row=get_global_id(0)*ROW_STEP;
    int col=get_global_id(1)*COL_STEP;
    float vec[5400];
    int index=0;
    for (int ch=0;ch<3;ch++){
        for (int y=0;y<60;y++){
            for (int x=0;x<30;x++){
                float value=band[((row+y)*band_cols+col+x)*3+ch]/255.;
                vec[index]=value;
                index++;
            }
        }
    }
    index=(row/ROW_STEP*col_num+col/COL_STEP)*5400;
    for (int i=0;i<5400;i++){
        vecs[index+i]=vec[i];
    }
    index=(row/ROW_STEP*col_num+col/COL_STEP)*4;
    float center_x=col*scale+15*scale;
    float center_y=row*scale+top+30*scale;
    pos[index]=center_x - 12.5*scale;
    pos[index+1]=center_y - 25*scale;
    pos[index+2]=center_x + 12.5*scale;
    pos[index+3]=center_y + 25*scale;
}