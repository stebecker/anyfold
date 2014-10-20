((c++-mode . (
         (eval . (progn
                   (defun my-project-specific-function ()
		     (setq ac-clang-cflags (
					    append ac-clang-cflags '("-I/home/steinbac/development/anyfold/include" "-I/home/steinbac/development/sqeazy/tests" )
						   )
		     	   )
		     (ac-clang-update-cmdlineargs)
                     )
		   (my-project-specific-function)
		   )
	       )
	 )
))
