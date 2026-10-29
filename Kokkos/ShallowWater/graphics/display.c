/*
 *  Copyright (c) 2011, Los Alamos National Security, LLC.
 *  All rights Reserved.
 *
 *  Copyright 2011. Los Alamos National Security, LLC. This software was produced 
 *  under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National 
 *  Laboratory (LANL), which is operated by Los Alamos National Security, LLC 
 *  for the U.S. Department of Energy. The U.S. Government has rights to use, 
 *  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS 
 *  ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
 *  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 *  to produce derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Los Alamos National Security, LLC, Los Alamos 
 *       National Laboratory, LANL, the U.S. Government, nor the names of its 
 *       contributors may be used to endorse or promote products derived from 
 *       this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE LOS ALAMOS NATIONAL SECURITY, LLC AND 
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
 *  NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 *  SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *  
 *  CLAMR -- LA-CC-11-094
 *  This research code is being developed as part of the 
 *  2011 X Division Summer Workshop for the express purpose
 *  of a collaborative code for development of ideas in
 *  the implementation of AMR codes for Exascale platforms
 *  
 *  AMR implementation of the Wave code previously developed
 *  as a demonstration code for regular grids on Exascale platforms
 *  as part of the Supercomputing Challenge and Los Alamos 
 *  National Laboratory
 *  
 *  Authors: Bob Robey       XCP-2   brobey@lanl.gov
 *           Neal Davis              davis68@lanl.gov, davis68@illinois.edu
 *           David Nicholaeff        dnic@lanl.gov, mtrxknight@aol.com
 *           Dennis Trujillo         dptrujillo@lanl.gov, dptru10@gmail.com
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "display.h"                                                                                                   
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#if defined(MINIMUM_PRECISION)
   typedef float real_t;
#elif defined(MIXED_PRECISION)
   typedef double real_t;
#elif defined(FULL_PRECISION)
   typedef double real_t;
#endif

#define ESCAPE 27
#ifdef HAVE_OPENGL
#define NCOLORS 5000
#else
#define NCOLORS 5000
#endif

#define WINSIZE 800 // 1200
int scale;
int DrawString(float x, float y, float z, char* string);
void InitGL(int width, int height);
void DrawSquares(void);
void DrawBoxes(void);
void SelectionRec(void);
void mouseClick(int button, int state, int x, int y);
void mouseDrag(int x, int y);
void keyPressed(unsigned char key, int x, int y);
void Scale();
void mpe_main_loop(void);
void display_get_event(void);

struct ColorTable {
   float Red;
   float Green;
   float Blue;
};

static int autoscale = 0;
static double display_circle_radius=-1.0;
static int current_frame = 0;
static double sim_time;
static int sim_cycle;

#ifdef HAVE_OPENGL
static struct ColorTable Rainbow[NCOLORS];
static int window;
#endif

#ifdef HAVE_OPENGL
// static real_t, xstart, ystart, xend, yend;
static int real_t, xstart, ystart, xend, yend;

#endif
enum mode_choice {EYE, MOVE, DRAW};
static int mode = MOVE;

static int width;
static float display_xmin=0.0, display_xmax=0.0, display_ymin=0.0, display_ymax=0.0;

#ifdef HAVE_OPENGL
static GLfloat xrot = 0.0, yrot = 0.0, xloc = 0.0, zloc = 0.0;
#else
static double xrot = 0.0, yrot = 0.0, xloc = 0.0, zloc = 0.0;
#endif

static int display_outline;
static int display_view_mode = 0;
static int display_mysize    = 0;

enum spatial_data_type {SPATIAL_DOUBLE, SPATIAL_FLOAT};
static int spatial_type = SPATIAL_FLOAT;

static double *x_double=NULL, *y_double=NULL, *dx_double=NULL, *dy_double=NULL;
static float *x_float=NULL, *y_float=NULL, *dx_float=NULL, *dy_float=NULL;

enum plot_data_type {DATA_DOUBLE, DATA_FLOAT};
static int data_type = DATA_FLOAT;
static double *data_double=NULL;
static float *data_float=NULL;
static int *display_proc=NULL;

int DrawString(float x, float y, float z, char* string) {
#ifdef HAVE_OPENGL
   char *c;
   glColor3f(0.0f, 0.0f, 0.0f);
   glRasterPos3f(x, y, z);
   for(c = string; *c != '\0'; c++) {
      glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *c);
      //glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
   }
#endif
#ifdef HAVE_OPENGL
   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- x %f y %f z %f string %s\n",x,y,z,string);
#endif
   return 1;
}

// function for x,y placement of text based
// on OpenGL window size data
// the ratio is an x or y offset from the
// top right corner of the window
double text_position(double coord_max,
                     double coord_min,
                     double ratio) {
   return coord_max - ((coord_max - coord_min) * ratio);
}

#ifdef HAVE_OPENGL
void InitGL(int width, int height) {
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
   glDepthFunc(GL_LESS);
   glShadeModel(GL_SMOOTH);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- width %d height %d\n",width,height);
}
#endif
void init_display(int *argc, char **argv, const char *title){

#ifdef HAVE_OPENGL
   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- argc %d argv %s title %s\n",*argc,argv[0],title);
#endif

   width = (WINSIZE / (display_ymax - display_ymin)) * (display_xmax - display_xmin);
#ifdef HAVE_OPENGL
   glutInit(argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(width, WINSIZE);
   glutInitWindowPosition(100, 100);

   window = glutCreateWindow(title);
#endif

   Scale();

#ifdef HAVE_OPENGL
   glutDisplayFunc(&draw_scene);

   glutKeyboardFunc(&keyPressed);
   glutMouseFunc(&mouseClick);
   glutMotionFunc(&mouseDrag);
   InitGL(width, WINSIZE);
#endif
}

void set_idle_function(void (*function)(void)){
#ifdef HAVE_OPENGL
   glutIdleFunc(function);
#endif

#ifdef HAVE_OPENGL
   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- function %p\n",function);
#endif
}

void start_main_loop(void){
#ifdef HAVE_OPENGL
   glutMainLoopEvent();
#endif
}
   

// function prototype for Fortran subroutine that
// will assign values for simulation time and
// cycle number
void provide_sim_progress(double simTime, int ncycle) {
   sim_time = simTime;
   sim_cycle = ncycle;
}

void set_display_window(float display_xmin_in, float display_xmax_in,
                        float display_ymin_in, float display_ymax_in){
   display_xmin = display_xmin_in;
   display_xmax = display_xmax_in;
   display_ymin = display_ymin_in;
   display_ymax = display_ymax_in;
}
void set_display_cell_data_double(double *data_in){
   data_type = DATA_DOUBLE;
   data_double = data_in;
}
void set_display_cell_data_float(float *data_in){
   data_type = DATA_FLOAT;
   data_float = data_in;
}
void set_display_cell_proc(int *display_proc_in){
   display_proc = display_proc_in;
}
void set_display_cell_coordinates_double(double *x_in, double *dx_in, double *y_in, double *dy_in){
   spatial_type = SPATIAL_DOUBLE;
   x_double = x_in;
   dx_double = dx_in;
   y_double = y_in;
   dy_double = dy_in;
}
void set_display_cell_coordinates_float(float *x_in, float *dx_in, float *y_in, float *dy_in){
   spatial_type = SPATIAL_FLOAT;
   x_float = x_in;
   dx_float = dx_in;
   y_float = y_in;
   dy_float = dy_in;
}
void free_display(void){
#ifdef HAVE_OPENGL
   glutDestroyWindow(window);
#endif
}
void DisplayState(void) {
   double scaleMax = 60.0, scaleMin = 0.0; //1200
   int i;
#ifdef HAVE_OPENGL
   int color;
#endif
   //vector<real_t> &H = state->H;

   if (autoscale) {
      scaleMax=-1.0e30;
      scaleMin=1.0e30;
      if (data_type == DATA_DOUBLE){
         for(i = 0; i<display_mysize; i++) {
            if (data_double[i] > scaleMax) scaleMax = data_double[i];
            if (data_double[i] < scaleMin) scaleMin = data_double[i];
         }
      } else {
         for(i = 0; i<display_mysize; i++) {
            if (data_float[i] > scaleMax) scaleMax = data_float[i];
            if (data_float[i] < scaleMin) scaleMin = data_float[i];
         }
      }
   }

#ifdef HAVE_OPENGL
   double step = NCOLORS/(scaleMax - scaleMin);
#endif
  
#ifdef HAVE_OPENGL
   //set up 2D viewing
   gluOrtho2D(display_xmin, display_xmax, display_ymin, display_ymax);
#endif
  
#ifdef HAVE_OPENGL
   for(i = 0; i < display_mysize; i++) {
      /*Draw filled cells with color set by state value*/
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glBegin(GL_QUADS);

      if (data_type == DATA_DOUBLE){
         color = (int)(data_double[i]-scaleMin)*step;
         if (DEBUG) printf("DEBUG color[%d] %d %lf %lf\n",i,color,data_double[i],step);
      } else {
         color = (int)(data_float[i]-scaleMin)*step;
      }
      if (color < 0) {
         color=0;
      }
      if (color >= NCOLORS) color = NCOLORS-1;
   
      glColor3f(Rainbow[color].Red, Rainbow[color].Green, Rainbow[color].Blue);
   
      if (spatial_type == SPATIAL_DOUBLE){
         if (DEBUG) printf("DEBUG draw vertex[%d] %lf %lf\n",i,x_double[i]-0.5*dx_double[i],y_double[i]-0.5*dy_double[i]);
         glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
         if (DEBUG) printf("DEBUG draw vertex[%d] %lf %lf\n",i,x_double[i]+0.5*dx_double[i],y_double[i]-0.5*dy_double[i]);
         glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
         if (DEBUG) printf("DEBUG draw vertex[%d] %lf %lf\n",i,x_double[i]+0.5*dx_double[i],y_double[i]+0.5*dy_double[i]);
         glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
         if (DEBUG) printf("DEBUG draw vertex[%d] %lf %lf\n",i,x_double[i]-0.5*dx_double[i],y_double[i]+0.5*dy_double[i]);
         glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
         if (DEBUG) printf("\n");
      } else {
         glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
         glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
         glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
         glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
      }
      glEnd();
   
      /*Draw cells again as outline*/
      if (display_outline) {
         glColor3f(0.0f, 0.0f, 0.0f);
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         glBegin(GL_QUADS);
         if (spatial_type == SPATIAL_DOUBLE){
            glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
            glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
            glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
            glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
         } else {
            glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
            glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
            glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
            glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
         }
         glEnd();
      }
   }
    
    // Render current frame number in OpenGL window
    char iteration_text[200];
    sprintf(iteration_text, "frame: %i", current_frame++);

    // Place the text in upper right corner of window
    // Use percentage offsets of total window sizes
    // for more adaptable placements

    DrawString (text_position(display_xmax,
                              display_xmin,
                              0.90),
                text_position(display_ymax,
                              display_ymin,
                              0.98),
                0, iteration_text);

    // Display the actual simulation iteration
    // number and time values below frame number
    // in graphics window
    char sim_time_text[200];
    char sim_cycle_text[200];
    sprintf(sim_time_text, "sim time (s): %g", sim_time);
    sprintf(sim_cycle_text, "sim cycle: %i", sim_cycle);
    DrawString (text_position(display_xmax,
                              display_xmin,
                              0.35),
                text_position(display_ymax,
                              display_ymin,
                              0.98),
                0, sim_time_text);
    DrawString (text_position(display_xmax,
                              display_xmin,
                              0.65),
                text_position(display_ymax,
                              display_ymin,
                              0.98),
                0, sim_cycle_text);

#endif

  
#ifdef HAVE_OPENGL
   glColor3f(0.0f, 0.0f, 0.0f);

   /* Draw circle for initial conditions */
   if (display_circle_radius > 0.0) {
      double radians;
      glBegin(GL_LINE_LOOP);
      for (double angle = 0.0; angle <= 360.0; angle+=1.0) {
         radians=angle*3.14159/180.0;
         glVertex2f(sin(radians) * display_circle_radius,cos(radians) * display_circle_radius);
      }
      glEnd();
   }
#endif
}

void DrawSquares(void) {
   if (display_proc == NULL) return;
#ifdef HAVE_OPENGL
   int i, color;
   int step = NCOLORS/(display_proc[display_mysize-1]+1);
#endif
   //step utilizes whole range of color, assumes last element of proc is last processor

#ifdef HAVE_OPENGL
   gluOrtho2D(display_xmin, display_xmax, display_ymin, display_ymax);
   //set up 2D viewing

   for(i = 0; i < display_mysize; i++) {
      /*Draw filled cells with color set by processor it is assigned to*/
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glBegin(GL_QUADS);
         color = display_proc[i]*step;
         glColor3f(Rainbow[color].Red, Rainbow[color].Green, Rainbow[color].Blue);

         if (spatial_type == SPATIAL_DOUBLE){
            glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
            glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
            glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
            glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
         } else {
            glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
            glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
            glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
            glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
         }
      glEnd();

      /*Draw cells again as outline*/
      glColor3f(0.0f, 0.0f, 0.0f);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glBegin(GL_QUADS);
         if (spatial_type == SPATIAL_DOUBLE) {
            glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
            glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]-0.5*dy_double[i]);
            glVertex2f(x_double[i]+0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
            glVertex2f(x_double[i]-0.5*dx_double[i], y_double[i]+0.5*dy_double[i]);
         } else {
            glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
            glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]-0.5*dy_float[i]);
            glVertex2f(x_float[i]+0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
            glVertex2f(x_float[i]-0.5*dx_float[i], y_float[i]+0.5*dy_float[i]);
         }
      glEnd();
   }
#endif

#ifdef HAVE_OPENGL
   /* Draw circle for initial conditions */
   if (display_circle_radius > 0.0) {
      double radians;
      glBegin(GL_LINE_LOOP);
      for (double angle = 0.0; angle <= 360.0; angle+=1.0) {
         radians=angle*3.14159/180.0;
         glVertex2f(sin(radians) * display_circle_radius,cos(radians) * display_circle_radius);
      }
      glEnd();
   }
#endif
  
#ifdef HAVE_OPENGL
   /*Trace order of cells with line going from center to center*/
   glBegin(GL_LINE_STRIP);
      if (spatial_type == SPATIAL_DOUBLE) {
         for(i = 0; i < display_mysize; i++) {
            glVertex2f(x_double[i]+dx_double[i]/2, y_double[i]+dy_double[i]/2);
         }
      } else {
         for(i = 0; i < display_mysize; i++) {
            glVertex2f(x_float[i]+dx_float[i]/2, y_float[i]+dy_float[i]/2);
         }
      }
   glEnd();
#endif
}

void DrawBoxes(void) {

#ifdef HAVE_OPENGL
   int i, color, step = NCOLORS/(display_proc[display_mysize-1]+1);
   //step utilizes whole range of color, assumes last element of proc is last processor

   glFrustum(display_xmin-1, display_xmax, display_ymin-1, display_ymax, 5.0, 10.0);
   glTranslatef(0.0f, 0.0f, -6.0f); //must move into screen to draw

   if (spatial_type == SPATIAL_DOUBLE){
      for(i = 0; i < display_mysize; i++) {
         /*Draw filled cells with color set by processor*/
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         glBegin(GL_QUADS);
            color = display_proc[i]*step;
            glColor3f(Rainbow[color].Red, Rainbow[color].Green, Rainbow[color].Blue);

            /*Front Face*/
            glVertex3f(x_double[i],              y_double[i],              0.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i],              0.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i]+dy_double[i], 0.0f);
            glVertex3f(x_double[i],              y_double[i]+dy_double[i], 0.0f);
            /*Right Face*/
            glVertex3f(x_double[i]+dx_double[i], y_double[i],              0.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i]+dy_double[i], 0.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i]+dy_double[i],-1.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i],             -1.0f);
            /*Back Face*/
            glVertex3f(x_double[i]+dx_double[i], y_double[i],             -1.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i]+dy_double[i],-1.0f);
            glVertex3f(x_double[i],              y_double[i]+dy_double[i],-1.0f);
            glVertex3f(x_double[i],              y_double[i],             -1.0f);
            /*Left Face*/
            glVertex3f(x_double[i],              y_double[i],              0.0f);
            glVertex3f(x_double[i],              y_double[i],             -1.0f);
            glVertex3f(x_double[i],              y_double[i]+dy_double[i],-1.0f);
            glVertex3f(x_double[i],              y_double[i]+dy_double[i], 0.0f);
            /*Bottom*/
            glVertex3f(x_double[i],              y_double[i],              0.0f);
            glVertex3f(x_double[i],              y_double[i],             -1.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i],             -1.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i],              0.0f);
            /*Top*/
            glVertex3f(x_double[i],              y_double[i]+dy_double[i], 0.0f);
            glVertex3f(x_double[i],              y_double[i]+dy_double[i],-1.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i]+dy_double[i],-1.0f);
            glVertex3f(x_double[i]+dx_double[i], y_double[i]+dy_double[i], 0.0f);
         glEnd();
      }
   } else {
      for(i = 0; i < display_mysize; i++) {
         /*Draw filled cells with color set by processor*/
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         glBegin(GL_QUADS);
            color = display_proc[i]*step;
            glColor3f(Rainbow[color].Red, Rainbow[color].Green, Rainbow[color].Blue);

            /*Front Face*/
            glVertex3f(x_float[i],             y_float[i],             0.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i],             0.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i]+dy_float[i], 0.0f);
            glVertex3f(x_float[i],             y_float[i]+dy_float[i], 0.0f);
            /*Right Face*/
            glVertex3f(x_float[i]+dx_float[i], y_float[i],             0.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i]+dy_float[i], 0.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i]+dy_float[i],-1.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i],            -1.0f);
            /*Back Face*/
            glVertex3f(x_float[i]+dx_float[i], y_float[i],            -1.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i]+dy_float[i],-1.0f);
            glVertex3f(x_float[i],             y_float[i]+dy_float[i],-1.0f);
            glVertex3f(x_float[i],             y_float[i],            -1.0f);
            /*Left Face*/
            glVertex3f(x_float[i],             y_float[i],             0.0f);
            glVertex3f(x_float[i],             y_float[i],            -1.0f);
            glVertex3f(x_float[i],             y_float[i]+dy_float[i],-1.0f);
            glVertex3f(x_float[i],             y_float[i]+dy_float[i], 0.0f);
            /*Bottom*/
            glVertex3f(x_float[i],             y_float[i],             0.0f);
            glVertex3f(x_float[i],             y_float[i],            -1.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i],            -1.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i],             0.0f);
            /*Top*/
            glVertex3f(x_float[i],             y_float[i]+dy_float[i], 0.0f);
            glVertex3f(x_float[i],             y_float[i]+dy_float[i],-1.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i]+dy_float[i],-1.0f);
            glVertex3f(x_float[i]+dx_float[i], y_float[i]+dy_float[i], 0.0f);
         glEnd();
      }
   }
#endif
}



void set_display_viewmode(int display_view_mode_in){
   display_view_mode = display_view_mode_in;
}
void set_display_mysize(int display_mysize_in){
   display_mysize = display_mysize_in;
}
void set_display_circle_radius(double display_circle_radius_in){
   display_circle_radius = display_circle_radius_in;
}
void set_display_outline(int display_outline_in){
   display_outline = display_outline_in;
}

void draw_scene(void) {
#ifdef HAVE_OPENGL
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glLoadIdentity();
#endif


   if (display_view_mode == 0) {
      DrawSquares();
   } else {
      DisplayState();
   }

   if (display_mysize <=500) {
      char c[10];
      if (data_type == DATA_DOUBLE){
         for(int i = 0; i < display_mysize; i++) {
            sprintf(c, "%d", i);
            DrawString(x_double[i]+0.5*dx_double[i], y_double[i]+0.5*dy_double[i], 0.0, c);
         }
      } else {
         for(int i = 0; i < display_mysize; i++) {
            sprintf(c, "%d", i);
            DrawString(x_float[i]+0.5*dx_float[i], y_float[i]+0.5*dy_float[i], 0.0, c);
         }
      }
   }

#ifdef HAVE_OPENGL
   if(mode == DRAW) {
      SelectionRec();
   }

   glLoadIdentity();

   glutSwapBuffers();
   glutMainLoopEvent();
#endif

   //if (display_mysize <= 500) sleep(1);
}


#ifdef HAVE_OPENGL
void SelectionRec(void) {
   glColor3f(0.0f, 0.0f, 0.0f);
   glLineWidth(2.0f);
   glBegin(GL_QUADS);
      glVertex2f(xstart, ystart);
      glVertex2f(xstart, yend);
      glVertex2f(xend, yend);
      glVertex2f(xend, ystart);
   glEnd();
   glLineWidth(1.0f);
}
#endif

void mouseClick(int button, int state, int x, int y) {
#ifdef HAVE_OPENGL
   if(state == GLUT_DOWN) {
      mode = EYE;
      xstart = (display_xmax-display_ymin)*(x/width)+display_ymin;
      ystart = (display_ymax-display_ymin)*((float)(WINSIZE-y)/WINSIZE)+display_ymin;
   }
   if(state == GLUT_UP) {
      glutPostRedisplay();
      mode = DRAW;
      xend = (display_xmax-display_ymin)*(x/width)+display_ymin;
      yend = (display_xmax-display_ymin)*((float)(WINSIZE-y)/WINSIZE)+display_ymin;
   }
   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- button %d\n",button);
#else
   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- x %d y %d button %d state %d\n",x,y,button,state);
#endif
}
void mouseDrag(int x, int y) {
#ifdef HAVE_OPENGL
   glutPostRedisplay();
   mode = DRAW;
   xend = (display_xmax-display_ymin)*(x/width)+display_ymin;
   yend = (display_xmax-display_ymin)*((float)(WINSIZE-y)/WINSIZE)+display_ymin;
#else
   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- x %d y %d\n",x,y);
#endif
}

void keyPressed(unsigned char key, int x, int y) {

    //usleep(100);

    if(key == ESCAPE) {
       free_display();
       exit(0);
    }
    if(key == 'm')   { mode = MOVE; }
    if(key == 'e')   { mode = EYE; }
    if(mode == EYE) {
      if(key == 'o') { xrot -= 5.0; }
      if(key == 'l') { xrot += 5.0; }
      if(key == 'k') { yrot -= 5.0; }
      if(key == ';') { yrot += 5.0; }
    }
    if(mode == MOVE) {
      if(key == 'o') { zloc += 1.0; }
      if(key == 'l') { zloc -= 1.0; }
      if(key == 'k') { xloc += 1.0; }
      if(key == ';') { xloc -= 1.0; }
    }

   // Just to get rid of compiler warnings
   if (1 == 2) printf("DEBUG -- x %d y %d\n",x,y);
}
void Scale() {
#ifdef HAVE_OPENGL
   int i;
   double r;
   for (i=0, r=0.0; i<200; i++, r+=.005) {
         Rainbow[i].Red = 0.0;
         Rainbow[i].Green = r;
         Rainbow[i].Blue = 1.0;
   }
   for (i=0, r=1.0; i<200; i++, r-=.005) {
         Rainbow[200+i].Red = 0.0;
         Rainbow[200+i].Green = 1.0;
         Rainbow[200+i].Blue = r;
   }
   for (i=0, r=0.0; i<200; i++, r+=.005) {
         Rainbow[400+i].Red = r;
         Rainbow[400+i].Green = 1.0;
         Rainbow[400+i].Blue = 0.0;
   }
   for (i=0, r=1.0; i<200; i++, r-=.005) {
         Rainbow[600+i].Red = 1.0;
         Rainbow[600+i].Green = r;
         Rainbow[600+i].Blue = 0.0;
   }
   for (i=0, r=0.0; i<200; i++, r+=.005) {
         Rainbow[800+i].Red = 1.0;
         Rainbow[800+i].Green = 0.0;
         Rainbow[800+i].Blue = r;
   }
#endif

}

/********************************************************************************/
void display_get_event(void)
/********************************************************************************/
{
   return;
}
