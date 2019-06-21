from easylib.log import TimedRotatingLoggerInitializer

name = 'dssm'

initializer = TimedRotatingLoggerInitializer(name='dssm', path='/tmp/dssm.log')
initializer.initialize()
