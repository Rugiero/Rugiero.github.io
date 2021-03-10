%% tsnovel.cls
%% Copyright 2017 Typesetters.se
%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either version 1.3
% of this license or (at your option) any later version.
% The latest version of this license is in
%   http://www.latex-project.org/lppl.txt
% and version 1.3 or later is part of all distributions of LaTeX
% version 2005/12/01 or later.
%
% This is a free LaTeX novel template from Typesetters.se.
% See novel.tex for an example of how to use this class.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tsnovel}[2017/01/06 Free LaTeX novel template from Typesetters.se]
\LoadClass[12pt, onecolumn, titlepage, twoside, openany]{book}

% Packages
\RequirePackage{blindtext}
\RequirePackage{color}
\RequirePackage{titlesec}
\RequirePackage{calligra}
\RequirePackage{lettrine}
\RequirePackage{fancyhdr}
\RequirePackage[utf8]{inputenc}

% Font Garamond
\RequirePackage[cmintegrals,cmbraces]{newtxmath}
\RequirePackage{ebgaramond-maths}
\RequirePackage[T1]{fontenc}

% Colors
\definecolor{steelblue}{RGB}{70,130,180}

% Chapter title styling
\renewcommand{\thechapter}{\Roman{chapter}}
\titleformat{\chapter}[display]
            {\bfseries\Large}
            {\filcenter\itshape{\chaptertitlename} \large
              \thechapter}
            {4ex}
            {\color{steelblue}\titlerule[2pt]
              \color{black}\Huge
             \vspace{1.5ex}\filcenter}
            [\vspace{1.5ex}
              \color{steelblue}\titlerule\thispagestyle{empty}]

% Header and footer styling
\fancyhf{}
\fancyhead[LE,RO]{\bfseries\thepage}
\fancyhead[CE]{\bfseries\MakeUppercase{\@ifundefined{@tstitle}{Book Title}{\@tstitle}}}
\fancyhead[CO]{\bfseries{\leftmark}}
\renewcommand{\headrulewidth}{1pt}
\renewcommand{\footrulewidth}{0pt}
\pagestyle{fancy}
\renewcommand{\headrule}{{\color{steelblue}
\hrule width\headwidth height\headrulewidth \vskip-\headrulewidth}}

% Chapter and section styling
\renewcommand{\chaptermark}[1]%
{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]%
             {\markright{\MakeUppercase{\thesection.\ #1}}}

% Command for typesetting preamble: title, copyright page and table of contents
\newcommand{\tsnovel}[6]{

\pagestyle{empty}

\def\@tstitle{#1}
\def\@tsauthor{#2}
\def\@tsyear{#3}
\def\@tsisbn{#4}
\def\@tspublished{#5}
\def\@tsprinted{#6}

\begin{center}
  \null
  \smallskip

  \Huge
  \textsc{\@tstitle}

  \bigskip
  \smallskip

  \Large
  by\, \textsc{\@tsauthor}
\end{center}

\pagebreak

\null
\vfill

\begin{center}
  Copyright \copyright \ \@tsyear \ by \@tsauthor \\

  All rights reserved. No part of this work may be reproduced or transmitted
  in any form or by any means, electronic or mechanical, including photocopying
  and recording, or by any information storage or retrieval system, without the
  prior written permission of the publisher.

  \bigskip

  ISBN: \@tsisbn

  Printed by \@tspublished \\
  Printed in \@tsprinted
\end{center}

\tableofcontents

\cleardoublepage
\pagestyle{fancy}
\thispagestyle{empty}

}

% Command for typesetting first letter of each chapter
\newcommand{\first}[1]{\lettrine{\color{steelblue}\calligra #1}{\, \, \, \,}}
