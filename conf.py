parallel_logging_configuration = {
        'version': 1,
        'formatters': {
                'default': {
                    'class': 'logging.Formatter',
                    'format': '%(asctime)s %(levelname)s %(name)s %(message)s'},
                'model': {
                    'class': 'logging.Formatter',
                    'format': '%(process)d %(levelname)s %(name)s %(funcName)s %(message)s'},
        },
        'handlers': {
                'console': {
                        'class' : 'logging.StreamHandler',
                        'formatter': 'model',
                        'level': 'INFO'  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                },
                'parallelAPES_file': {
                        'class': 'logging.FileHandler',
                        'level': 'INFO',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'formatter': 'default',
                        'filename': 'parallelAPES.log',
                        'mode': 'w',  # a == append, w == overwrite
                },
        },
        'loggers': {
                'pyAPES': {
                        #'handlers': ['file'],
                        'level': 'INFO',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'propagate': True,
                        },
                'canopy':{
                        #'handlers': ['file'],
                        'level': 'INFO',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'propagete': True,
                        },
                'soil':{
                        #'handlers': ['file'],
                        'level': 'INFO',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'propagete': True,
                },
        },
        'root': {
                'level': 'INFO',
                'handlers': ['console', 'parallelAPES_file']
        }
    }