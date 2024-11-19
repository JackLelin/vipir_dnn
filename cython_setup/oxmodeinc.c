#include <stdio.h>
#include <math.h>

int oxmodeinc(  //input
                int zlen, 
                int folen,
                int fxlen,
                double * fps, 
                double * Z,
                double * fes,
                double fcO,
                double fcX,
                // output
                double * fvsO,
                double * vhsO,
                double * fvsX,
                double * vhsX,
		double theta)
{
    double dz = Z[1] - Z[0];
    double fv, vh, n2b, fp, z, X, n2t, prev_dh, prevn2b;
    // O mode
    for(int fidx=0; fidx<folen; fidx++) // for all ionosonde frequencies
    {
        fv = fvsO[fidx]; // indexing ionosonde frequency
        if(fv < fcO+0.025)
        {
            vh = 0;
            n2b = 1;
            for(int zidx=0; zidx<zlen; zidx++) //for all height
            {
                fp = fps[zidx]; //indexing the height and the fps
                z = Z[zidx];
                X = (fp/fv)*(fp/fv);
                n2t = 1-X;
                if(n2t>0)
                {
                    prev_dh = dz*((1/sqrt(n2b)+1/sqrt(n2t))/2.);
                    prevn2b = n2b;
                    vh += prev_dh;
                    n2b = n2t;
                }
                else
                {
                    // substract last dh and add last slab made of two slabs:
                    vh=vh - prev_dh + 2*(2*dz)*sqrt(prevn2b)/(prevn2b-n2t) *(1/(sin(theta)*sin(theta)));
                    break;
                }
            }
            vhsO[fidx] = vh;
        }
        else break;
    }
    // X-trace forward model
    for(int fidx=0; fidx<fxlen; fidx++) // for all ionosonde frequencies
    {
        double fv = fvsX[fidx];
        double vh, n2b, fxb, fp, z, fe, X, Y, n2t, fxt, prev_dh, prevn2b;
        if(fv<fcX+0.025)
        {
            vh = 0; 
            n2b = 1; 
            fxb = 1;
            for(int zidx=0; zidx<zlen; zidx++)
            {
                fp = fps[zidx]; //indexing the height and the fps
                z = Z[zidx];
                fe = fes[zidx];
                X = (fp/fv)*(fp/fv);
                Y = fe/fv;
                n2t = 1-X*(1-X)/(1-X-Y*Y);
                fxt = n2t+X*((1-X)/(1-X-Y*Y))*((1-X)/(1-X-Y*Y))+(X*Y/(1-X-Y*Y))*(X*Y/(1-X-Y*Y));
                if(n2t>0)
                {
                    prev_dh = dz/((sqrt(n2b)/fxb+sqrt(n2t)/fxt)/2.); // keep track of latest dh
                    prevn2b = n2b;                                     // keep track of latest nb2
                    vh += prev_dh;
                    n2b=n2t;
                    fxb=fxt;
                }
                else
                {
                    vh=vh - prev_dh + 2*(2*dz)*sqrt(prevn2b)/(prevn2b-n2t)*(2-Y)/(1-Y)*(1/(1+cos(theta)*cos(theta)));
                    break;
                }
            }
        vhsX[fidx] = vh;
        }
        else break;
    }
    return 1;
}
