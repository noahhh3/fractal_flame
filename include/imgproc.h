

#ifndef IMGPROC_H
#define IMGPROC_H

#include <string>
#include <vector>

struct Point { float x; float y; };

namespace img
{

class ColorLUT
{
  public:
    ColorLUT(double gamma = 1.0);
    ~ColorLUT(){}

    void operator()(const double& value, std::vector<float>& C) const;

  private:
    double gamma;

    std::vector<float> black;
    //vector of vector(which holds rgb values)
    std::vector<std::vector<float>> bands;
};


class ImgProc
{

  public:

    //! Construct with no content
    ImgProc();
   ~ImgProc();

    //! delete existing content and leave in a blank state
    void clear();
    //! delete existing content and re-initialize to the input dimensions with value 0.0
    void clear(int nX, int nY, int nC);

    //! Load an image from a file.  Deletes exising content.
    bool load( const std::string& filename );

    //! Retrieve the width
    int nx() const { return Nx; }
    //! Retrieve the height
    int ny() const { return Ny; }
    //! Retrieve the number of channels
    int depth() const { return Nc; }

    //! Retrieve the (multichannel) value at a pixel.  Copies the value into parameter 'pixel'.
    void value( int i, int j, std::vector<float>& pixel) const;
    //! Set the (multichannel) value at a pixel.
    void set_value( int i, int j, const std::vector<float>& pixel);



    /*Image statistics functions*/
    //finds the max value for a channel, param specifies what channel
    float max_channel_val(int);

    //find the min value for a channel, param specifies what channels
    float min_channel_val(int);

    //find the avg pixel channel value for a specific channel in the image
    float average_channel_val(int);

    //find channel stddev
    float channel_stddev(int);

    //display all statistics
    void display_stats();



    /* Histogram functions for an image */
    //cdf
    void cdf_histogram(std::vector<float>& cdf, std::vector<float>& pdf);

    //pdf
    void pdf_histogram(std::vector<int>&, std::vector<float>&);

    //fill in a histogram for a channel
    void create_histogram(int, std::vector<int>&);

    //histogram equalization
    void histogram_equalization();



    /*FRACTAL FLAME FUNCTIONS*/
    //initialize requirements to perform the ifs/fractal flame algoirthm
    void init_ifs();

    // variation definitons

    //use specific variations and number of iterations to perform fractal flame
    void fractal_flame(size_t, std::vector<void(*)(Point&)>&, ColorLUT&);




    //! Copy constructor. Clears existing content.
    ImgProc(const ImgProc& v);
    //! Copy assignment. Clears existing content.
    ImgProc& operator=(const ImgProc& v);

    friend void swap(ImgProc& u, ImgProc& v);

    //! multiplies all pixels and channels by a value
    void operator*=(float v);
    //! divides all pixels and channels by a value
    void operator/=(float v);
    //! adds a value to all pixels and channels
    void operator+=(float v);
    //! subtracts a value from all pixels and channels
    void operator-=(float v);

    //! converts image to its compliment in-place
    void compliment();
    //increase/decrease img brightness
    void brightness(float);
    //increase/decrease img bias
    void bias(float);
    //increase/decrease img gamma
    void gamma(float);
    //quantize image function
    void quantize(int);
    //grayscale image func
    void grayscale();
    //rms contrast function
    void rms_contrast();
    //! indexing to a particular pixel and channel
    long index(int i, int j, int c) const;
    //! indexing to a particular pixel
    long index(int i, int j) const;

    //writes image to a file
    void write_to_file();

    //! returns raw pointer to data (dangerous)
    float* raw(){ return img_data; }

  private:

    int Nx, Ny, Nc;

    bool ifs_started;

    long Nsize;
    float * img_data;
};
//! swaps content of two images
void swap(ImgProc& u, ImgProc& v);

}


Point Bent (Point);

#endif
