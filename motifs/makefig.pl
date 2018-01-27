#!/usr/bin/perl -w
use strict;
use Getopt::Long;

my ($right, $middle, $left, $not_squashed) = (0, 0, 0, 0);
my ($rightbar, $leftbar) = ("true", "true");
my $numbering;
my $startnum = 1;
GetOptions("left" => \$left, "right" => \$right, "middle" => \$middle, "from:s" => \$startnum, "nonumbers" => \$numbering, "not_squashed" => \$not_squashed);
$numbering = (defined $numbering) ? "false" : "true";

if ($left) {
    $rightbar = "false";
} elsif ($middle) {
    $rightbar = "false";
    $leftbar = "false";
} elsif ($right) {
    $leftbar = "false";
} elsif ($not_squashed) {
    $leftbar = "false";
}

my @nts = qw(A C G T);
my (%p, @h);

my $j = 0;
my @sum;
while (<>) {
    chomp;
    s/^\s*//g;
    my $n = $nts[$j];
    my @l = split (/\s+/, $_);
    for (my $i = 0; $i < @l; $i++) {
	$p{$n}[$i] = $l[$i];
	$sum[$i] += $l[$i];
    }
    $j++;
}
# norm
my $width = scalar(@{$p{'A'}});
for (my $i = 0; $i < $width; $i++) {
    for (my $j = 0; $j < @nts; $j++) {
	my $n = $nts[$j];
	$p{$n}[$i] /= $sum[$i] if ($sum[$i] > 0);
    }
}

my $dimensions;
{
    my $left = 40;
    my $pxl_width = $width * 43;
    my $right = $left + $pxl_width;
    $left -= 40 if ($leftbar eq "true");
    $right += 4 if ($rightbar eq "true");
    $dimensions = join(" ", $left, 0, $right, "377");
}

print eps_header();
for (my $i = 0; $i < @{$p{'A'}}; $i++) {
    $h[$i] = 2;
    foreach my $n (@nts) {
	$h[$i] += $p{$n}[$i] * log(1e-50+$p{$n}[$i]) / log(2);
    }
    $h[$i] = 2 if ($not_squashed);
    my $p = $startnum + $i;
    $p %= 10;
    print "% at coordinate $p\n";
    print "numbering {($p) makenumber} if\n";
    print "gsave\n";
    foreach my $k (sort {$p{$a}[$i] <=> $p{$b}[$i] } @nts) {
	my $v = $p{$k}[$i];
	$v *= $h[$i] * 5;
 	$v += 0.001 if ($v == 0);
	print "$v($k) numchar\n" if ($p{$k}[$i] > 0);
# 	print "$v($k) numchar\n" if ($p{$k}[$i] >= 0.05);
    }
    print "grestore\n";
    print "shift\n";
}
print eps_footer();

sub eps_header {
return <<"EOF";
%!PS-Adobe-2.0 EPSF-2.0
%%Title: PSSM
%%Creator: Tom Schneider, toms\@ncifcrf.gov
%%BoundingBox: $dimensions
%%Pages: atend
%%DocumentFonts:
%%EndComments
/llx  14.2 def
/lly  10.9 def
/urx 1063.0 def
/ury 793.7 def

/cmfactor 72 2.54 div def % defines points -> cm conversion
/cm {cmfactor mul} bind def % defines centimeters

% user defined parameters
/lowest 1 def
/highest 31 def
/bar 1 def
/xcorner  1.50000 cm def
/ycorner 2.50000 cm def
/rotation 0.00000 def % degrees
/charwidth  1.49355 cm def
/barheight 10.00000 cm def
/barwidth  0.10000 cm def
/barbits  2.00000 def % bits
/Ibeamfraction  1.00000 def
/barendsr $rightbar def
/barendsl $leftbar def
/showingbox false def
/outline false def
/caps true def
/stacksperline 31 def
/linesperpage 1 def
/linemove  3.30000 def
/numbering $numbering def
/shrinking false def
/edgecontrol ( ) def
/edgeleft  2.00000 def
/edgeright  2.00000 def
/edgelow  2.00000 def
/edgehigh  2.00000 def
/shrink  0.50000 def
/ShowEnds (-) def % d: DNA, p: PROTEIN, -: none
/HalfWhiteIbeam false def

/knirhs 1 shrink sub 2 div def
/charwidth4 charwidth 4 div def
/charwidth2 charwidth 2 div def

/setthelinewidth {1 setlinewidth} def
setthelinewidth % set to normal linewidth

% define fonts
/ffss {findfont fontsize scalefont setfont} def
/FontForStringRegular {/Times-Bold       ffss} def
/FontForStringItalic  {/Times-BoldItalic ffss} def
/FontForLogo          {/Helvetica-Bold   ffss} def
/FontForPrime         {/Symbol           ffss} def
/FontForSymbol        {/Symbol           ffss} def

% Set up the font size for the graphics
/fontsize charwidth def

% movements to place 5' and 3' symbols
/fivemovex {0} def
/fivemovey {(0) charparams lx ux sub 3 mul} def
/threemovex {(0) stringwidth pop 0.5 mul} def
/threemovey {fivemovey} def
/prime {FontForPrime (\242) show FontForStringRegular} def

% make italics possible in titles
/IT {% TRstring ITstring IT -
  exch show
  FontForStringItalic
  show
  FontForStringRegular
} def


% make symbols possible in titles
/SY {% TRstring SYstring SY -
  exch show
  FontForSymbol
  show
  FontForStringRegular
} def

%(*[[ This special comment allows deletion of the repeated
% procedures when several logos are concatenated together
% See the censor program.

/charparams { % char charparams => uy ux ly lx
% takes a single character and returns the coordinates that
% defines the outer bounds of where the ink goes
  gsave
    newpath
    0 0 moveto
    % take the character off the stack and use it here:
    true charpath 
    flattenpath 
    pathbbox % compute bounding box of 1 pt. char => lx ly ux uy
    % the path is here, but toss it away ...
  grestore
  /uy exch def
  /ux exch def
  /ly exch def
  /lx exch def
} bind def

/dashbox { % xsize ysize dashbox -
% draw a dashed box of xsize by ysize (in points)
  /ysize exch def % the y size of the box
  /xsize exch def % the x size of the box
  1 setlinewidth
  gsave
    % Define the width of the dashed lines for boxes:
    newpath
    0 0 moveto
    xsize 0 lineto
    xsize ysize lineto
    0 ysize lineto
    0 0 lineto
    [3] 0 setdash
    stroke
  grestore
  setthelinewidth
} bind def

/boxshow { % xsize ysize char boxshow
% show the character with a box around it, sizes in points
gsave
  /tc exch def % define the character
  /ysize exch def % the y size of the character
  /xsize exch def % the x size of the character
  /xmulfactor 1 def /ymulfactor 1 def

  % if ysize is negative, make everything upside down!
  ysize 0 lt {
    % put ysize normal in this orientation
    /ysize ysize abs def
    xsize ysize translate
    180 rotate
  } if

  shrinking {
    xsize knirhs mul ysize knirhs mul translate
    shrink shrink scale
  } if

  2 {
    gsave
    xmulfactor ymulfactor scale
    tc charparams
    grestore

    ysize % desired size of character in points
    uy ly sub % height of character in points
    dup 0.0 ne {
      div % factor by which to scale up the character
      /ymulfactor exch def
    } % end if
    {pop pop}
    ifelse

    xsize % desired size of character in points
    ux lx sub % width of character in points
    dup 0.0 ne {
      div % factor by which to scale up the character
      /xmulfactor exch def
    } % end if
    {pop pop}
    ifelse
  } repeat

  % Adjust horizontal position if the symbol is an I
  tc (I) eq {charwidth 2 div % half of requested character width
             ux lx sub 2 div % half of the actual character
                sub      0 translate} if
  % Avoid x scaling for I
  tc (I) eq {/xmulfactor 1 def} if

  /xmove xmulfactor lx mul neg def
  /ymove ymulfactor ly mul neg def

  newpath
  xmove ymove moveto
  xmulfactor ymulfactor scale

  tc show
grestore
} def

/numchar{ % charheight character numchar
% Make a character of given height in cm,
% then move vertically by that amount
  gsave
    /char exch def
    /charheight exch cm def
    char (N) eq {0 1 0 setrgbcolor} if
    char (Q) eq {0 1 0 setrgbcolor} if
    char (K) eq {0 0 1 setrgbcolor} if
    char (R) eq {0 0 1 setrgbcolor} if
    char (H) eq {0 0 1 setrgbcolor} if
    char (D) eq {1 0 0 setrgbcolor} if
    char (E) eq {1 0 0 setrgbcolor} if
    char (F) eq {1 1 0 setrgbcolor} if
    char (L) eq {1 1 0 setrgbcolor} if
    char (I) eq {1 1 0 setrgbcolor} if
    char (M) eq {1 1 0 setrgbcolor} if
    char (V) eq {1 1 0 setrgbcolor} if
    char (P) eq {1 0 1 setrgbcolor} if
    char (T) eq {0 0.7 0 setrgbcolor} if
    char (S) eq {1 0 1 setrgbcolor} if
    char (C) eq {0 0 1 setrgbcolor} if
    char (A) eq {1 0 0 setrgbcolor} if
    char (G) eq {1 0.6 0 setrgbcolor} if
    char (Y) eq {1 0 1 setrgbcolor} if
    char (W) eq {1 0 1 setrgbcolor} if
    char (-) eq {0 0 0 setrgbcolor} if
    charwidth charheight char boxshow
  grestore
  charheight abs 1 gt {0 charheight abs translate} if
} bind def

/Ibar{
% make a horizontal bar
gsave
  newpath
    charwidth4 neg 0 moveto
    charwidth4 0 lineto
  stroke
grestore
} bind def

/Ibeam{ % height Ibeam
% Make an Ibeam of twice the given height, in cm
  /height exch cm def
  /heightDRAW height Ibeamfraction mul def
  1 setlinewidth
     HalfWhiteIbeam outline not and
     {0.75 setgray} % grey on bottom
     {0 setgray} % black on bottom
  ifelse
  gsave
    charwidth2 height neg translate
    Ibar
    newpath
      0 0 moveto
      0 heightDRAW rlineto
    stroke
    0 setgray % black on top
    newpath
      0 height moveto
      0 height rmoveto
      currentpoint translate
    Ibar
    newpath
      0 0 moveto
      0 heightDRAW neg rlineto
      currentpoint translate
    stroke
  grestore
  setthelinewidth
} bind def

/makenumber { % number makenumber
% make the number
gsave
  shift % shift to the other side of the stack
  0 rotate % rotate so the number fits
  dup stringwidth pop % find the length of the number
  neg % prepare for move
  charwidth (0) charparams uy ly sub % height of numbers
  sub 2 div %
  moveto % move back to provide space
  /stringscale 1.8 def
  -1.45 cm -2.4 cm moveto
  stringscale stringscale scale
  show
grestore
} bind def

/shift{ % move to the next horizontal position
charwidth 0 translate
} bind def

/bar2 barwidth 2 div def
/bar2n bar2 neg def
/makebar { % make a vertical bar at the current location
gsave
   bar2n 0 moveto
   barwidth 0 rlineto
   0 barheight rlineto
   barwidth neg 0 rlineto
   closepath
   fill
grestore
} def

% definitions for maketic
/str 10 string def % string to hold number
% points of movement between tic marks:
% (abs protects against barbits being negative)
/ticmovement barheight barbits abs div def

/maketic { % make tic marks and numbers
gsave
  % initial increment limit proc for
  0 1 barbits abs cvi
  {/loopnumber exch def

    % convert the number coming from the loop to a string
    % and find its width
    loopnumber 10 str cvrs
    /stringnumber exch def % string representing the number

    stringnumber stringwidth pop
    /numberwidth exch def % width of number to show

    /halfnumberheight
      stringnumber charparams % capture sizes
      uy ly sub 2 div
    def

    numberwidth 1.5 mul % move back two digits
    neg loopnumber ticmovement mul % shift on y axis
    halfnumberheight sub % down half the digit

    moveto % move back the width of the string

    stringnumber show

    % now show the tic mark
    0 halfnumberheight rmoveto % shift up again
    numberwidth 0 rlineto
    stroke
  } for
  grestore
} def

/degpercycle 360 def
 
/cosine {%    amplitude  phase  wavelength  base
%             xmin ymin xmax ymax step dash thickness
%             cosine -
% draws a cosine wave with the given parameters:
% amplitude (points): height of the wave
% phase (points): starting point of the wave
% wavelength (points): length from crest to crest
% base (points): lowest point of the curve
% xmin ymin xmax ymax (points): region in which to draw
% step steps for drawing a cosine wave
% dash if greater than zero, size of dashes of the wave (points)
% thickness if greater than zero, thickness of wave (points)

  /thickness exch def
  /dash exch def
  /step exch def
  /ymax exch def
  /xmax exch def
  /ymin exch def
  /xmin exch def
  /base exch def
  /wavelength exch def
  /phase exch def
  /amplitude exch def
  % fun := amplitude*cos( ((-y-phase)/wavelength)*360) + base
  /fun {phase sub wavelength div degpercycle mul cos
           amplitude mul base add} def

  gsave
    /originallinewidth currentlinewidth def
    thickness 0 gt {thickness setlinewidth} if
    /c currentlinewidth def
%   Make the curve fit into the region specified
    newpath
    xmin ymin c sub moveto
    xmax ymin c sub lineto
    xmax ymax c add lineto
    xmin ymax c add lineto
    closepath
    clip

    newpath
    xmin dup fun moveto
    xmin step xmax { % loop from xmin by step to xmax
      dup fun lineto } for
    dash 0 gt {[dash cvi] 0 setdash} if % turn dash on
    stroke

    originallinewidth setlinewidth
  grestore
} bind def

/circlesymbol { % x y radius circlesymbol - (path)
newpath 0 360 arc closepath} bind def

/sqrt3 3 sqrt def
/trianglesymbol { % x y radius trianglesymbol - (path)
/r exch def
/sqrt3r sqrt3 r mul def
translate
120 rotate
0 r translate
-120 rotate
newpath
0 0 moveto
sqrt3r 0 lineto
-300 rotate
sqrt3r 0 lineto
closepath} bind def

/squaresymbol { % x y side squaresymbol - (path)
/side exch def
translate
side 2 div neg dup translate
newpath
0 0 moveto
0 side lineto
side side lineto
side 0 lineto
closepath} bind def

/linesymbol { % x1 y1 x2 y2 linesymbol - (path)
/y2 exch def
/x2 exch def
/y1 exch def
/x1 exch def
newpath
x1 y1 moveto
x2 y2 lineto
} bind def

/boxsymbol { % x1 y1 x2 y2 boxsymbol - (path)
/y2 exch def
/x2 exch def
/y1 exch def
/x1 exch def
newpath
x1 y1 moveto
x2 y1 lineto
x2 y2 lineto
x1 y2 lineto
closepath
} bind def

% The following special comment allows deletion of the repeated
% procedures when several logos are concatenated together
% See the censor program.
%]]%*)

/startpage { % start a page
  save % [ startpage
  % set the font used in the title strings
  FontForStringRegular
  gsave % [ startpage
  xcorner ycorner translate
  rotation rotate
  % create the user defined strings
  gsave
    /stringscale  1.00000 def
     0.00000 cm 15.00000 cm moveto
    stringscale stringscale scale
    ()
    show
  grestore
  gsave
    /stringscale  0.50000 def
     0.00000 cm -1.50000 cm moveto
    stringscale stringscale scale
    ()
    show
  grestore
  % now move up to the top of the top line:
  0 linesperpage linemove barheight mul mul translate

  % set the font used in the logos
  FontForLogo
} def

%(*[[ This special comment allows deletion of the repeated
% procedures when several logos are concatenated together
% See the censor program.

/endpage { % end a page
  grestore % ] endpage
  showpage % REMOVE FOR PACKAGING INTO ANOTHER FIGURE
  restore % ] endpage
} def

/showleftend {
gsave
 charwidth neg 0 translate
 fivemovex fivemovey moveto ShowEnds (d) eq {(5) show prime} if
 ShowEnds (p) eq {(N) show} if
grestore
} def

/showrightend {
gsave
 threemovex threemovey moveto ShowEnds (d) eq {(3) show prime} if
 ShowEnds (p) eq {(C) show} if
grestore
} def

/startline{ % start a line
% move down to the bottom of the line:
  0 linemove barheight mul neg translate
  gsave % [ startline
  barendsl {
    maketic % maketic.startline
    gsave
      bar2n 0 translate % makebar.startline
      makebar % makebar.startline
    grestore
  } if
  showleftend
} def

/endline{ % end a line
  showrightend
  barendsr {
    gsave
      bar2 0 translate % makebar.endline
      makebar % makebar.endline
    grestore
  } if
  grestore % ] startline
} def

% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% @@@@@@@@@@@@@@@@@@@@ End of procedures @@@@@@@@@@@@@@@@@@@
% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

% The following special comment allows deletion of the repeated
% procedures when several logos are concatenated together
% See the censor program.
%]]%*)

%%EndProlog

%%Page: 1 1
startpage % [
startline % line number 1

EOF
}

sub eps_footer {
return <<"EOF";
endline
% Rs total is 10.95755 +/-  0.32404 bits in the range from 1 to 31
%%Trailer
%%Pages: 1

EOF
}
