//------------------------------------------------
//
//  img_paint
//
//
//-------------------------------------------------




#include <cmath>
#include <omp.h>
#include "imgproc.h"
#include "CmdLineFind.h"
#include <vector>



#include <GL/gl.h>   // OpenGL itself.
#include <GL/glu.h>  // GLU support library.
#include <GL/glut.h> // GLUT support library.


#include <iostream>
#include <stack>


using namespace std;
using namespace img;

ImgProc image;




void setNbCores( int nb )
{
   omp_set_num_threads( nb );
}

void cbMotion( int x, int y )
{
}

void cbMouse( int button, int state, int x, int y )
{
}

void cbDisplay( void )
{
   glClear(GL_COLOR_BUFFER_BIT );
   glDrawPixels( image.nx(), image.ny(), GL_RGBA, GL_FLOAT, image.raw() );
   glutSwapBuffers();
}

void cbIdle()
{
   glutPostRedisplay();
}

void cbOnKeyboard( unsigned char key, int x, int y )
{

   /*some functions are commented out because they can be implemnted with
     the operator overloads*/
   float increase_amt, decrease_amt = 0.0;

   int steps = 0;

   switch (key)
   {

      //invert operatioin
      case 'c':
	      image.compliment();
	      cout << "Compliment\n";
	      break;

      //increase image brightness
      case 'V':
        increase_amt = 1.1;
        //image.brightness(increase_amt);
        image*=increase_amt;
        cout << "Increase brightness\n";
        break;

      //decrease image brightness
      case 'v':
        decrease_amt = 0.9;
        //image.brightness(decrease_amt);
        image*=decrease_amt;
        cout << "Decrease brightness\n";
        break;

      //increae overall image bias
      case 'B':
        increase_amt = 0.1;
        image.bias(increase_amt);
        cout << "Increase overall bias\n";
        break;

      //decrease overall image bias
      case 'b':
        decrease_amt = -0.1;
        image.bias(decrease_amt);
        cout << "Decrease overall bias\n";
        break;

      //increase image gamma
      case 'G':
        increase_amt = 1.1;
        image.gamma(increase_amt);
        //image.gamma(1.8);
        cout << "Increase gamma\n";
        break;

      //decrease image gamma
      case 'g':
        decrease_amt = 0.9;
        image.gamma(decrease_amt);
        cout << "Decrease gamma\n";
        break;

      //convert image to grayscale
      case 'w':
        image.grayscale();
        cout << "Grayscale\n";
        break;

      //quantize image with steps
      case 'q':
        steps = 5;
        image.quantize(steps);
        cout << "Quantize\n";
        break;

      //apply rms contrast to image
      case 'C':
        image.rms_contrast();
        cout << "Rms contrast\n";
        break;

      //histogram equalization independently to each channel
      case 'H':
        cout << "Applying Histogram equalization to each channel\n";
        image.histogram_equalization();
        break;

      //display image channel stats
      case 'S':
        cout << "Displaying Statistics:\n\n";
        image.display_stats();
        break;

      //run fractal flame algorithm / iterated function systems
      case '!':
        cout << "IFS / Fractal Flame\n\n";
        image.init_ifs();
        glutPostRedisplay();
        break;

      //write image to a file
      case 'o':
        image.write_to_file();
        cout << "Writing to file\n";
        break;
   }
}

void PrintUsage()
{
  cout << "img_paint keyboard choices\n";
  cout << "\n    Pixel Manipulation\n";
  cout << "c         compliment\n";
  cout << "V         increase brightness\n";
  cout << "v         decrease brightness\n";
  cout << "B         increase bias\n";
  cout << "b         decrease bias\n";
  cout << "G         increase gamma\n";
  cout << "g         decrease gamma\n";
  cout << "w         black and white\n";
  cout << "q         quantize\n";
  cout << "C         rms contrast\n";

  cout << "\n    Fractal Flame\n";
  cout << "!         IFS / Fractal Flame\n";

  cout << "\n    Statistics\n";
  cout << "S         display statistics\n";
  cout << "H         histogram equalization\n";

  cout << "\n    File I/O\n";
  cout << "o         write to file\n";
}


int main(int argc, char** argv)
{
   lux::CmdLineFind clf( argc, argv );

   setNbCores(8);

   string imagename = clf.find("-image", "", "Image to drive color");

   clf.usage("-h");
   clf.printFinds();
   PrintUsage();

   image.load(imagename);


   // GLUT routines
   glutInit(&argc, argv);

   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize( image.nx(), image.ny() );

   // Open a window
   char title[] = "img_paint";
   glutCreateWindow( title );

   glClearColor( 1,1,1,1 );

   glutDisplayFunc(&cbDisplay);
   glutIdleFunc(&cbIdle);
   glutKeyboardFunc(&cbOnKeyboard);
   glutMouseFunc( &cbMouse );
   glutMotionFunc( &cbMotion );

   glutMainLoop();
   return 1;
};
