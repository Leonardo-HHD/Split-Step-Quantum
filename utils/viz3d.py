import numpy as np
from mayavi import mlab
from tvtk.util import ctf

try:
    from .cmfunc import complex_to_rgb
except ImportError:
    from cmfunc import complex_to_rgb

def plot_wavefunc(Psi:np.ndarray, Lx, Ly, Lz, contrast_vals=[0.01,0.99], mode='surface'):

    mlab.figure(1, size=(800, 800), bgcolor=(0, 0, 0))

    abs_max= np.amax(np.abs(Psi))
    psi = (Psi)/(abs_max)
    amp = np.abs(psi)
    arg = np.angle(psi)


    if mode == 'volume':
        field = mlab.pipeline.scalar_field(amp)
        vol = mlab.pipeline.volume(field)

        # Change the color transfer function
        color1 = complex_to_rgb(np.exp( 1j*2*np.pi/10*0))
        color2 = complex_to_rgb(-np.exp( 1j*2*np.pi/10*0))
        c = ctf.save_ctfs(vol._volume_property)
        c['rgb'] = [[-0.45, *color1],
                    [-0.4, *color1],
                    [-0.3, *color1],
                    [-0.2, *color1],
                    [-0.001, *color1],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.001, *color2],
                    [0.2, *color2],
                    [0.3, *color2],
                    [0.4, *color2],
                    [0.45, *color2]]

        c['alpha'] = [[-0.5, 1.0],
                      [-contrast_vals[1], 1.0],
                      [-contrast_vals[0], 0.0],
                      [0, 0.0],
                      [contrast_vals[0], 0.0],
                      [contrast_vals[1], 1.0],
                      [0.5, 1.0]]
        ctf.load_ctfs(c, vol._volume_property)
        # Update the shadow LUT of the volume module.
        vol.update_ctf = True

        # colour_data = (arg.ravel())%(2*np.pi)
        # field.image_data.point_data.add_array(colour_data)
        # field.image_data.point_data.get_array(1).name = 'phase'
        # field.update()
        # field2 = mlab.pipeline.set_active_attribute(field,
        #                                             point_scalars='scalar')
        # contour = mlab.pipeline.contour(field2)
        # contour.filter.contours= [isovalue,]
        # contour2 = mlab.pipeline.set_active_attribute(contour,
        #                                             point_scalars='phase')
        # s = mlab.pipeline.surface(contour2, colormap='hsv', vmin= 0.0 ,vmax= 2.*np.pi)

        # s.scene.light_manager.light_mode = 'vtk'
        # s.actor.property.interpolation = 'phong'

    elif mode == 'abs-volume':

        amp = np.where(amp > contrast_vals[1], contrast_vals[1],amp)
        amp = np.where(amp < contrast_vals[0], contrast_vals[0],amp)
        field = mlab.pipeline.scalar_field(amp)
        vol = mlab.pipeline.volume(field)

        # Update the shadow LUT of the volume module.
        vol.update_ctf = True

    elif mode == 'surface':

        isovalue = np.mean(contrast_vals)

        field = mlab.pipeline.scalar_field(amp)

        arr = mlab.screenshot(antialiased = False)

        colour_data = (arg.T.ravel())%(2*np.pi)
        field.image_data.point_data.add_array(colour_data)
        field.image_data.point_data.get_array(1).name = 'phase'
        field.update()
        field2 = mlab.pipeline.set_active_attribute(field,
                                                    point_scalars='scalar')
        contour = mlab.pipeline.contour(field2)
        contour.filter.contours= [isovalue,]
        contour2 = mlab.pipeline.set_active_attribute(contour,
                                                    point_scalars='phase')
        s = mlab.pipeline.surface(contour2, colormap='hsv', vmin= 0.0 ,vmax= 2.*np.pi)

        s.scene.light_manager.light_mode = 'vtk'
        s.actor.property.interpolation = 'gouraud' #'flat','gouraud','phong'
        # s.actor.property.specular = 0.5
        # s.actor.property.specular_power = 20
        # s.actor.property.smooth_shading = True

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'volume', 'abs-volume' or 'surface'.")

    mlab.outline()
    mlab.axes(xlabel='x [Å]', ylabel='y [Å]', zlabel='z [Å]', nb_labels=6 , ranges = (-Lx/2,Lx/2,-Ly/2,Ly/2,-Lz/2,Lz/2))

    #azimuth angle
    φ = 30
    N = int((sum(np.array(Psi.shape)**2)/3)**0.5)
    mlab.view(azimuth= φ,  distance=N*3.5)
    mlab.show()

if __name__ == '__main__':
    # Example usage
    Lx, Ly, Lz = 10, 10, 10
    x,y,z=np.mgrid[-Lx/2:Lx/2:40j, -Ly/2:Ly/2:40j, -Lz/2:Lz/2:40j]
    r=(x**2+y**2)**0.5
    t=np.arctan2(y,x)
    Psi = (np.exp(-0.5*r**2-0.2*z**2))*(np.sin(1*r**2)**2)*(np.sin(3*z)**2)*(np.exp(1j*t*z))

    mlab.close(all=True)
    plot_wavefunc(Psi, Lx, Ly, Lz, contrast_vals=[0.01,0.99], mode='volume')
    mlab.close(all=True)