#ifndef VIDEO
#define VIDEO
#include "video.h"
#endif // !VIDEO

Video::Video(const char *prefix)
{
	char *filename = new char[strlen(prefix) + 7];
	char temp_name[40];

	frame_length = 0;

	for (int i = 0; i < 2; i++)
	{
		sprintf(temp_name, "_%d.bmp", i);
		strcpy(filename, prefix);
		strcat(filename, temp_name);
		frame[i] = imread(filename);
		cvtColor(frame[i], gray[i], CV_RGB2GRAY);
		GaussianBlur(gray[i], gray[i], Size(15, 15), 1.5, 1.5);

		frame_length++;
	}
	
	delete []filename;
}