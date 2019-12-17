set nocompatible              " required
filetype off                  " required

filetype plugin on
set omnifunc=syntaxcomplete#Complete

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'gmarik/Vundle.vim'

" add all your plugins here (note older versions of Vundle
" used Bundle instead of Plugin)

" ...

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required

" Enable folding
set foldmethod=indent
set foldlevel=99

syntax on

" Enable folding with the spacebar
nnoremap <space> za

" Get python indentation right
au BufNewFile,BufRead *.py 
	\setlocal tabstop=4 
    \softtabstop=4 
	\shiftwidth=4 
	\textwidth=79 
	\expandtab autoindent 
	\fileformat=unix

" Highlight searches
set hlsearch

" Hilight mode for bad white space
:highlight BadWhitespace ctermbg=darkgreen guibg=darkgreen

" Get rid of bad white space
au BufRead,BufNewFile *.py,*.pyw,*.c,*.h match BadWhitespace /\s\+$/

" Use UTF-8 because of portuguese
set encoding=utf-8

" My own definitions for plain vi, when vim is not available
" source .exrc
