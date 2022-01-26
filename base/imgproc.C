
#include <cmath>
#include "imgproc.h"


#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING

using namespace img;




/* COLOR LUT FUNCTIONS */
ColorLUT::ColorLUT(double gam) : gamma(gam)
{

  std::vector<float> C;
  C.push_back(0.0);
  C.push_back(0.0);
  C.push_back(0.0);
  black = C;


  C[0] = 4.0/255.0;
  C[1] = 6.0/255.0;
  C[2] = 75.0/255.0;
  bands.push_back(C);

  C[0] = 150.0/255.0;
  C[1] = 79.0/255.0;
  C[2] = 76.0/255.0;
  bands.push_back(C);

  C[0] = 120.0/255.0;
  C[1] = 30.0/255.0;
  C[2] = 40.0/255.0;
  bands.push_back(C);

  C[0] = 70.0/255.0;
  C[1] = 10.0/255.0;
  C[2] = 30.0/255.0;
  bands.push_back(C);

  C[0] = 70.0/255.0;
  C[1] = 80.0/255.0;
  C[2] = 165.0/255.0;
  bands.push_back(C);

  C[0] = 255.0/255.0;
  C[1] = 60.0/255.0;
  C[2] = 60.0/255.0;
  bands.push_back(C);

}

void ColorLUT::operator()(const double& value, std::vector<float>& C) const
{

  C = black;
  if(value > 1.0 || value < 0.0) return;

  double x = std::pow(value, gamma) * (bands.size()-1);

  size_t low_index = (size_t)x;
  size_t high_index = low_index + 1;

  double weight = x * (double)low_index;

  if(high_index >= bands.size()) { high_index = bands.size()-1; }

  for(size_t c=0; c<C.size(); c++) {
    C[c] = bands[low_index][c] * (1.0-weight) + bands[high_index][c] * weight;
  }
}






/* IMGPROC FUNCTIONS */
ImgProc::ImgProc() :
  Nx (0),
  Ny (0),
  Nc (0),
  ifs_started(0),
  Nsize (0),
  img_data (nullptr)
{}

ImgProc::~ImgProc()
{
   clear();
}

void ImgProc::clear()
{
   if( img_data != nullptr ){ delete[] img_data; img_data = nullptr;}
   Nx = 0;
   Ny = 0;
   Nc = 0;
   Nsize = 0;
}

void ImgProc::clear(int nX, int nY, int nC)
{
   clear();
   Nx = nX;
   Ny = nY;
   Nc = nC;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i] = 0.0; }
}

bool ImgProc::load( const std::string& filename )
{
   auto in = ImageInput::create (filename);
   if (!in) {return false;}
   ImageSpec spec;
   in->open (filename, spec);
   clear();
   Nx = spec.width;
   Ny = spec.height;
   Nc = spec.nchannels;
   ifs_started = false;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
   in->read_image(TypeDesc::FLOAT, img_data);
   in->close ();
   return true;
}


void ImgProc::value( int i, int j, std::vector<float>& pixel) const
{
   pixel.clear();
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   pixel.resize(Nc);
   for( int c=0;c<Nc;c++ )
   {
      pixel[c] = img_data[index(i,j,c)];
   }
   return;
}

void ImgProc::set_value( int i, int j, const std::vector<float>& pixel)
{
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   if( Nc > (int)pixel.size() ){ return; }
#pragma omp parallel for
   for( int c=0;c<Nc;c++ )
   {
      img_data[index(i,j,c)] = pixel[c];
   }
   return;
}



ImgProc::ImgProc(const ImgProc& v) :
  Nx (v.Nx),
  Ny (v.Ny),
  Nc (v.Nc),
  Nsize (v.Nsize)
{
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
}

ImgProc& ImgProc::operator=(const ImgProc& v)
{
   if( this == &v ){ return *this; }
   if( Nx != v.Nx || Ny != v.Ny || Nc != v.Nc )
   {
      clear();
      Nx = v.Nx;
      Ny = v.Ny;
      Nc = v.Nc;
      ifs_started = false;
      Nsize = v.Nsize;
   }
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
   return *this;
}



/*STATISTICS FUNCTIONS*/

//find channel max value
float ImgProc::max_channel_val(int c)
{
  float max_val=0;
  //loop through specified channel
  for (long i=c; i<Nsize; i+=Nc) {
    max_val = std::max(img_data[i], max_val);
  }

  return max_val;
}


//find channel min value
float ImgProc::min_channel_val(int c)
{
  float min_val=0;
  for (long i=c; i<Nsize; i+=Nc) {
    min_val = std::min(img_data[i], min_val);
  }

  return min_val;
}


// find the average value for a channel
float ImgProc::average_channel_val(int c)
{
  float mean=0;
  for (long i=c; i<Nsize; i+=Nc) {
    mean += img_data[i];
  }

  //we do Nx*Ny and not Nsize because Nx and Ny is the size of the picture
  //it is how many pixels we have, where as Nsize stores each individual
  //channel, so we want to make sure that we aren't including all pixel channels
  return (mean/(float)(Nx*Ny));
}


// find the standard deviation of a channel
float ImgProc::channel_stddev(int c)
{
  float mean = average_channel_val(c);
  float stddev = 0;

  //loop through specific channel
  for (long i=c; i<Nsize; i+=Nc) {
    stddev += (pow(img_data[i] - mean, 2));
  }

  stddev = sqrt(((stddev)/(float)(Nx*Ny)));

  return stddev;
}


//display stats for all image channels
void ImgProc::display_stats()
{
  //print out stats for each channel
  for(int i=0; i<Nc; i++) {
    std::cout << "Channel " << i << " min value: "
              << min_channel_val(i) << std::endl;

    std::cout << "Channel " << i << " max value: "
              << max_channel_val(i) << std::endl;

    std::cout << "Channel " << i << " avg value: "
              << average_channel_val(i) << std::endl;

    std::cout << "Channel " << i << " stddev value: "
                        << channel_stddev(i) << std::endl;

    std::cout << "\n\n";
  }
}



/*HISTOGRAM FUNCTIONS*/

//create a cdf distribution
void ImgProc::cdf_histogram(std::vector<float>& cdf, std::vector<float>& pdf)
{
  //cdf keeps track of area under curve up until certain point
  cdf[0] = pdf[0];

  for(unsigned int i=1; i<cdf.size(); i++) {
    cdf[i] = cdf[i-1] + pdf[i];
  }
}


//create a probability density function for a histogram
void ImgProc::pdf_histogram(std::vector<int>& histogram,std::vector<float>& pdf)
{
  for(unsigned int i=0; i<histogram.size(); i++) {
    pdf[i] = ((float)(histogram[i])/(float)(Nx*Ny));
    //std::cout << histogram[i] << std::endl;
  }
}


//create a histogram for an image color channel
void ImgProc::create_histogram(int c, std::vector<int>& histogram)
{

  float i_max = max_channel_val(c);
  float i_min = min_channel_val(c);
  float delta_i = (i_max-i_min)/(histogram.size());

  for (long i=c; i<Nsize; i+=Nc) {
    int m = (int)((img_data[i] - i_min)/(delta_i));
    //increase count of intesnity for that bin m
    histogram[m]++;
  }
}


//apply histogram equalization to an image
void ImgProc::histogram_equalization()
{
  //int num val range for a color channel
  int N = 255;

  //basically the black and white points how much of a channel is present
  //in the image
  std::vector<std::vector<int>> channel_histograms(Nc, std::vector<int>(N,0));
  std::vector<std::vector<float>> channel_pdf(Nc, std::vector<float>(N,0.0));
  std::vector<std::vector<float>> channel_cdf(Nc, std::vector<float>(N,0.0));

  std::vector<float> i_max(Nc,0);
  std::vector<float> i_min(Nc,1);

  for(int i=0; i<Nc; i++) {
    create_histogram(i, channel_histograms[i]);
    //pdf/cdf histograms
    pdf_histogram(channel_histograms[i],channel_pdf[i]);
    cdf_histogram(channel_cdf[i], channel_pdf[i]);
    i_max[i] = max_channel_val(i);
    i_min[i] = min_channel_val(i);
  }

  for(int i=0; i<Nx; i++) {
    for(int j=0; j<Ny; j++) {
      for(int c=0; c<Nc; c++) {
        float delta_i = (i_max[c]-i_min[c])/(float)(N);
        float Q = ((img_data[index(i,j,c)]-i_min[c])/delta_i);
        int q = Q;

        //weight value
        float w = Q-q;

        //determine how to play intensity equalization
        if(q<N-1) {
          img_data[index(i,j,c)] = (channel_cdf[c][q]*(1-w)) +
                                    (channel_cdf[c][q+1] * w);
        }
        else if(q==N-1) {
          img_data[index(i,j,c)] = channel_cdf[c][q];
        }
      }
    }
  }
}


void ImgProc::operator*=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] *= v; }
}

void ImgProc::operator/=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] /= v; }
}

void ImgProc::operator+=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] += v; }
}

void ImgProc::operator-=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] -= v; }
}


//channel not matter 1-Pi is applied to all
void ImgProc::compliment()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] = 1.0 - img_data[i]; }
}

void ImgProc::brightness(float bx)
{
  if (img_data == nullptr) return;
  for (long i=0; i<Nsize;i++) img_data[i]*=bx;
}

//overall image bias
void ImgProc::bias(float B)
{
  if (img_data == nullptr) return;
  //loop through area of pixels
  for (long i=0; i<Nsize; i++) img_data[i] += B;

}

void ImgProc::gamma(float power)
{
  if (img_data == nullptr) return;
  for (long i=0; i<Nsize; i++) img_data[i] = pow(img_data[i],power);
}

void ImgProc::quantize(int N)
{
  if (img_data == nullptr) return;
  for (long i=0; i<Nsize; i++) img_data[i] = ((int)(img_data[i]*N)/(float)N);
}

void ImgProc::grayscale()
{
  if(img_data == nullptr) return;
  for (int i=0; i<Nx; i++) {
    for (int j=0; j<Ny; j++) {
      float g_avg=0;
      for (int c=0; c<Nc; c++) {
        long xyc_index = index(i,j,c);
        if(c==0) g_avg += 0.2126 * img_data[xyc_index];
        else if(c==1) g_avg += 0.7152 * img_data[xyc_index];
        else if(c==2) g_avg += 0.0722 * img_data[xyc_index];
      }
      for (int channel=0; channel<Nc; channel++) {
	      long xyc_index = index(i,j,channel);
        img_data[xyc_index] = g_avg;
      }
    }
  }
}

void ImgProc::rms_contrast()
{
  if(img_data == nullptr) return;

  //holds sum of pixels squared for a channel
  float* channel_squared = new float[Nc];

  //holds the mean for pixel of a certain color channel
  float* channel_average = new float[Nc];

  //G value
  float* contrast_channel = new float[Nc];

  if (channel_squared == nullptr) return;
  if (channel_average == nullptr) return;
  if (contrast_channel == nullptr) return;

  //calculate channel means
  // order N time
  for (int i=0; i<Nx; i++) {
    for (int j=0; j<Ny; j++) {
      for (int c=0; c<Nc; c++) {
        long xyc = index(i,j,c);
        channel_squared[c] += pow(img_data[xyc],2);
        channel_average[c] += img_data[xyc];
      }
    }
  }

  //std dev
  for (int c=0; c<Nc; c++) {
    channel_average[c] = (channel_average[c]/(float)(Nx*Ny));
    contrast_channel[c] = sqrt(((channel_squared[c])/(Nx*Ny))
                          - pow(channel_average[c],2));
  }

  //assign each contrasted value
  for (int i=0; i<Nx; i++) {
    for (int j=0; j<Ny; j++) {
      for (int c=0; c<Nc; c++) {
        long xyc = index(i,j,c);
        img_data[xyc] = (float)((img_data[xyc]-channel_average[c])
                         /contrast_channel[c]);
      }
    }
  }

  //free allocated memory
  delete[] channel_squared;
  delete[] contrast_channel;
  delete[] channel_average;

  channel_squared = contrast_channel = channel_average = nullptr;

}





/* ITERATED FUNCTIONS SYSTEMS */
// ifs function definitions
void bent(Point &p)
{
  if((p.x < 0) && (p.y >= 0)) {
    p.x = p.x * 2;
    return;
  }
  if((p.x >= 0) && (p.y < 0)) {
    p.y = p.y / 2;
    return;
  }
  if((p.x < 0) && (p.y < 0)) {
    p.x = p.x * 2;
    p.y = p.y / 2;
    return;
  }
}
void polar(Point &p)
{
  float theta = atan2(p.x,p.y);
  float r = pow((pow(p.x,2) + pow(p.y,2)), 0.5);
  p.x = ((theta/M_PI));
  p.y = (r-1);
}
void fisheye(Point &p)
{
  float r = pow((pow(p.x,2) + pow(p.y,2)), 0.5);
  p.x = (2/(r+1)) * p.y;
  p.y = (2/(r+1)) * p.x;
}
void diamond(Point &p)
{
  float r = sqrt((pow(p.x,2) + pow(p.y,2)));
  float theta = atan2(p.x,p.y);
  p.x = sin(theta) * cos(r);
  p.y = cos(theta) * sin(r);
}
void tangent(Point &p)
{
  p.x = (sin(p.x)/cos(p.y));
  p.y = tan(p.y);
}
void spiral(Point &p)
{
  float r = sqrt((pow(p.x,2) + pow(p.y,2)));
  float theta = atan2(p.x,p.y);
  p.x = sin(theta)/r;
  p.y = sin(theta) - cos(r);
}
void hyperbolic(Point &p)
{
  float r = sqrt((pow(p.x,2) + pow(p.y,2)));
  float theta = atan2(p.x,p.y);
  p.x = sin(theta)/r;
  p.y = r*cos(theta);
}
//initialize data for an ifs algoirthm
void ImgProc::init_ifs()
{
  if(!ifs_started) {
    unsigned int width = 1920, height = 1080, channels = 4;
    // has 4 channels, r,g,b and alpha
    clear(width,height, channels);
    ifs_started = true;
  }

  void (*f0)(Point&) = polar;
  void (*f1)(Point&) = hyperbolic;
  void (*f2)(Point&) = tangent;
  void (*f3)(Point&) = diamond;

  std::vector<void(*)(Point&)> var;

  ColorLUT color_lut;

  var.push_back(f0);
  var.push_back(f1);
  var.push_back(f2);
  var.push_back(f3);

  size_t iterations = 500000;
  fractal_flame(iterations, var, color_lut);

}
//fractal flame algorithm
void ImgProc::fractal_flame(size_t iterations, std::vector<void(*)(Point&)> &variations, ColorLUT &lut)
{
  Point p;
  //drand48 generates random number b/w 0 and 1, the subtract one,makes it
  // -1 to 1
  p.x = 2.0*drand48() - 1.0;
  p.y = 2.0*drand48() - 1.0;

  float w =  drand48();

  std::vector<float>func_weights(variations.size());
  func_weights[0] = 0.45;
  func_weights[1] = 0.1;
  func_weights[2] = 0.1;
  func_weights[3] = 0.25;

  for(size_t iter=0; iter<iterations; iter++) {

    //find index for our library of variations
    size_t func_index = (size_t) (drand48() * variations.size());
    (*variations[func_index])(p);
    w = (w+ func_weights[func_index]) * 0.5;

    //start plotting after first 20 iterations
    if(iter > 20) {
      if(p.x >= -1.0 && p.x <= 1.0 && p.y >= -1.0 && p.y <= 1.0) {

        //re index x and y values
        float x = p.x + 1.0;
        float y = p.y + 1.0;
        x *= 0.5 * Nx;
        y *= 0.5 * Ny;
        int i = x;
        if(i<Nx) {
          int j = y;
          if(j<Ny) {

            std::vector<float> color;
            lut(w,color);

            //gets the pixel rgba values at the current i,j index
            std::vector<float> pc;
            value(i,j,pc);

            for(size_t c=0; c<pc.size()-1; c++) {
              pc[c] = pc[c] * pc[pc.size()-1];
              pc[c] = (pc[c] + color[c]) / (pc[pc.size()-1] + 1);
            }
            pc[pc.size()-1]+=1;
            set_value(i,j,pc);
          }
        }
      }
    }
  }
}






void ImgProc::write_to_file()
{
  //std::string file_name = "modified.jpeg";
  std::string file_name = "modified.exr";
  ImageOutput* out = ImageOutput::create(file_name);
  if(out==nullptr) return;
  ImageSpec spec(Nx, Ny, Nc, TypeDesc::FLOAT);
  out->open(file_name, spec);
  out->write_image(TypeDesc::FLOAT, img_data);
  ImageOutput::destroy(out);
}


long ImgProc::index(int i, int j, int c) const
{
   return (long) c + (long) Nc * index(i,j); // interleaved channels

   // return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
}

long ImgProc::index(int i, int j) const
{
   return (long) i + (long)Nx * (long)j;
}



void img::swap(ImgProc& u, ImgProc& v)
{
   float* temp = v.img_data;
   int Nx = v.Nx;
   int Ny = v.Ny;
   int Nc = v.Nc;
   long Nsize = v.Nsize;

   v.Nx = u.Nx;
   v.Ny = u.Ny;
   v.Nc = u.Nc;
   v.Nsize = u.Nsize;
   v.img_data = u.img_data;

   u.Nx = Nx;
   u.Ny = Ny;
   u.Nc = Nc;
   u.Nsize = Nsize;
   u.img_data = temp;
}
