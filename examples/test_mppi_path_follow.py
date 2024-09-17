import time
import math

from robotics_algorithm.env.two_d_world import TwoDWorldDiffDrive
from robotics_algorithm.control.mppi import MPPI

# This path is computed using Hybrid A*
PATH_DT = 0.1
shortest_path = [
    [0.5, 0.5, 0],
    [0.5499914331243889, 0.5006544237076969, 0.05235987755983018],
    [0.5998801048365815, 0.5039243000069666, 0.10471975511966036],
    [0.6495292737551336, 0.5098006663941593, 0.15707963267949054],
    [0.6988028549591109, 0.5182674161574689, 0.20943951023932073],
    [0.747565792987398, 0.5293013425243106, 0.2617993877991509],
    [0.7956844320163294, 0.5428722022694717, 0.3141592653589811],
    [0.8430268822010127, 0.5589427986096899, 0.36651914291881127],
    [0.8894633811762249, 0.5774690831574498, 0.41887902047864145],
    [0.9348666497260364, 0.5984002766545524, 0.47123889803847163],
    [0.9791122406472946, 0.6216790081545315, 0.5235987755983018],
    [1.0224135108365164, 0.6466790081545316, 0.5235987755983018],
    [1.0657147810257381, 0.6716790081545317, 0.5235987755983018],
    [1.1090160512149598, 0.6966790081545319, 0.5235987755983018],
    [1.1523173214041815, 0.721679008154532, 0.5235987755983018],
    [1.1956185915934032, 0.7466790081545321, 0.5235987755983018],
    [1.238919861782625, 0.7716790081545323, 0.5235987755983018],
    [1.2822211319718466, 0.7966790081545324, 0.5235987755983018],
    [1.3255224021610683, 0.8216790081545325, 0.5235987755983018],
    [1.36882367235029, 0.8466790081545327, 0.5235987755983018],
    [1.4121249425395117, 0.8716790081545328, 0.5235987755983018],
    [1.4557460054506717, 0.8961079771610232, 0.47123889803847163],
    [1.5005858006641273, 0.9182205170747194, 0.41887902047864145],
    [1.5465214254179744, 0.9379560189607417, 0.36651914291881127],
    [1.593426973357709, 0.9552603891802186, 0.3141592653589811],
    [1.6411738796367927, 0.9700861976571905, 0.2617993877991509],
    [1.6896312733039613, 0.9823928078811512, 0.20943951023932073],
    [1.7386663360114054, 0.992146488288899, 0.15707963267949054],
    [1.788144666060629, 0.9993205047204061, 0.10471975511966036],
    [1.8379306467881642, 1.0038951936952936, 0.05235987755983018],
    [1.8878878182814176, 1.005858016309064, 0.0],
    [1.9378792514058065, 1.0065124400167609, 0.05235987755983018],
    [1.9877679231179992, 1.0097823163160307, 0.10471975511966036],
    [2.0374170920365513, 1.0156586827032232, 0.15707963267949054],
    [2.0866906732405286, 1.0241254324665328, 0.20943951023932073],
    [2.1354536112688156, 1.0351593588333745, 0.2617993877991509],
    [2.183572250297747, 1.0487302185785357, 0.3141592653589811],
    [2.23091470048243, 1.0648008149187536, 0.36651914291881127],
    [2.2773511994576423, 1.0833270994665134, 0.41887902047864145],
    [2.322754468007454, 1.104258292963616, 0.47123889803847163],
    [2.367000058928712, 1.127537024463595, 0.5235987755983018],
    [2.4099666981321755, 1.1530994885814936, 0.575958653158132],
    [2.4515366170463615, 1.1808756203799902, 0.6283185307179622],
    [2.491595875413016, 1.2107892874125201, 0.6806784082777924],
    [2.530034673589441, 1.2427584983970208, 0.7330382858376225],
    [2.566747653501683, 1.2766956279483364, 0.7853981633974527],
    [2.60163418742369, 1.3125076567533076, 0.8377580409572829],
    [2.634598653790916, 1.3500964265302433, 0.8901179185171131],
    [2.66555069929238, 1.3893589090739487, 0.9424777960769433],
    [2.6944054865228124, 1.430187488648873, 0.9948376736367734],
    [2.7210839265160867, 1.4724702569563612, 1.0471975511966036],
    [2.746083926516086, 1.5157715271455834, 1.0471975511966036],
    [2.7710839265160856, 1.5590727973348055, 1.0471975511966036],
    [2.796083926516085, 1.6023740675240277, 1.0471975511966036],
    [2.8210839265160845, 1.6456753377132498, 1.0471975511966036],
    [2.846083926516084, 1.688976607902472, 1.0471975511966036],
    [2.8710839265160835, 1.732277878091694, 1.0471975511966036],
    [2.896083926516083, 1.7755791482809162, 1.0471975511966036],
    [2.9210839265160824, 1.8188804184701384, 1.0471975511966036],
    [2.946083926516082, 1.8621816886593605, 1.0471975511966036],
    [2.9710839265160813, 1.9054829588485827, 1.0471975511966036],
    [2.9966463906339795, 1.9484495980520462, 0.9948376736367734],
    [3.024422522432476, 1.9900195169662325, 0.9424777960769433],
    [3.0543361894650056, 2.030078775332887, 0.8901179185171131],
    [3.0863054004495063, 2.068517573509312, 0.8377580409572829],
    [3.120242530000821, 2.105230553421554, 0.7853981633974527],
    [3.156054558805792, 2.1401170873435618, 0.7330382858376225],
    [3.1936433285827275, 2.173081553710788, 0.6806784082777924],
    [3.2329058111264324, 2.204033599212252, 0.6283185307179622],
    [3.2737343907013567, 2.232888386442685, 0.575958653158132],
    [3.3160171590088448, 2.2595668264359596, 0.5235987755983018],
    [3.358983798212308, 2.2851292905538587, 0.575958653158132],
    [3.400553717126494, 2.3129054223523555, 0.6283185307179622],
    [3.4406129754931487, 2.3428190893848853, 0.6806784082777924],
    [3.4790517736695734, 2.374788300369386, 0.7330382858376225],
    [3.5157647535818155, 2.4087254299207017, 0.7853981633974527],
    [3.5506512875038228, 2.4445374587256725, 0.8377580409572829],
    [3.5836157538710487, 2.4821262285026084, 0.8901179185171131],
    [3.6145677993725127, 2.5213887110463133, 0.9424777960769433],
    [3.643422586602945, 2.5622172906212377, 0.9948376736367734],
    [3.6701010265962193, 2.6045000589287257, 1.0471975511966036],
    [3.6956634907141175, 2.647466698132189, 0.9948376736367734],
    [3.723439622512614, 2.689036617046375, 0.9424777960769433],
    [3.7533532895451436, 2.7290958754130297, 0.8901179185171131],
    [3.7853225005296443, 2.767534673589455, 0.8377580409572829],
    [3.819259630080959, 2.804247653501697, 0.7853981633974527],
    [3.85507165888593, 2.8391341874237046, 0.7330382858376225],
    [3.8926604286628654, 2.872098653790931, 0.6806784082777924],
    [3.9319229112065703, 2.903050699292395, 0.6283185307179622],
    [3.9727514907814947, 2.931905486522828, 0.575958653158132],
    [4.015034259088982, 2.9585839265161025, 0.5235987755983018],
    [4.058335529278204, 2.983583926516103, 0.5235987755983018],
    [4.101636799467426, 3.008583926516103, 0.5235987755983018],
    [4.144938069656647, 3.0335839265161035, 0.5235987755983018],
    [4.188239339845869, 3.058583926516104, 0.5235987755983018],
    [4.231540610035091, 3.0835839265161042, 0.5235987755983018],
    [4.274841880224312, 3.1085839265161046, 0.5235987755983018],
    [4.318143150413534, 3.133583926516105, 0.5235987755983018],
    [4.361444420602756, 3.1585839265161053, 0.5235987755983018],
    [4.4047456907919775, 3.1835839265161057, 0.5235987755983018],
    [4.448046960981199, 3.208583926516106, 0.5235987755983018],
    [4.491348231170421, 3.2335839265161064, 0.5235987755983018],
    [4.534649501359643, 3.2585839265161067, 0.5235987755983018],
    [4.577950771548864, 3.283583926516107, 0.5235987755983018],
    [4.621252041738086, 3.3085839265161074, 0.5235987755983018],
    [4.664553311927308, 3.333583926516108, 0.5235987755983018],
    [4.707854582116529, 3.358583926516108, 0.5235987755983018],
    [4.751155852305751, 3.3835839265161085, 0.5235987755983018],
    [4.794457122494973, 3.408583926516109, 0.5235987755983018],
    [4.8377583926841945, 3.433583926516109, 0.5235987755983018],
    [4.881059662873416, 3.4585839265161096, 0.5235987755983018],
    [4.924360933062638, 3.48358392651611, 0.5235987755983018],
    [4.96766220325186, 3.5085839265161103, 0.5235987755983018],
    [5.010963473441081, 3.5335839265161106, 0.5235987755983018],
    [5.054264743630303, 3.558583926516111, 0.5235987755983018],
    [5.097566013819525, 3.5835839265161114, 0.5235987755983018],
    [5.140867284008746, 3.6085839265161117, 0.5235987755983018],
    [5.184168554197968, 3.633583926516112, 0.5235987755983018],
    [5.22746982438719, 3.6585839265161124, 0.5235987755983018],
    [5.2707710945764115, 3.6835839265161128, 0.5235987755983018],
    [5.314072364765633, 3.708583926516113, 0.5235987755983018],
    [5.357373634954855, 3.7335839265161135, 0.5235987755983018],
    [5.4006749051440766, 3.758583926516114, 0.5235987755983018],
    [5.443976175333298, 3.783583926516114, 0.5235987755983018],
    [5.48727744552252, 3.8085839265161145, 0.5235987755983018],
    [5.530578715711742, 3.833583926516115, 0.5235987755983018],
    [5.573879985900963, 3.8585839265161153, 0.5235987755983018],
    [5.617181256090185, 3.8835839265161156, 0.5235987755983018],
    [5.660482526279407, 3.908583926516116, 0.5235987755983018],
    [5.703783796468628, 3.9335839265161163, 0.5235987755983018],
    [5.74708506665785, 3.9585839265161167, 0.5235987755983018],
    [5.7900517058613135, 3.9841463906340158, 0.575958653158132],
    [5.831621624775499, 4.011922522432513, 0.6283185307179622],
    [5.871680883142154, 4.041836189465043, 0.6806784082777924],
    [5.910119681318578, 4.073805400449543, 0.7330382858376225],
    [5.94683266123082, 4.107742530000858, 0.7853981633974527],
    [5.981719195152827, 4.143554558805829, 0.8377580409572829],
    [6.014683661520054, 4.181143328582764, 0.8901179185171131],
    [6.045635707021517, 4.2204058111264695, 0.9424777960769433],
    [6.0744904942519495, 4.261234390701393, 0.9948376736367734],
    [6.101168934245224, 4.303517159008881, 1.0471975511966036],
    [6.126168934245224, 4.346818429198103, 1.0471975511966036],
    [6.1511689342452245, 4.390119699387324, 1.0471975511966036],
    [6.176168934245225, 4.433420969576546, 1.0471975511966036],
    [6.201168934245225, 4.476722239765768, 1.0471975511966036],
    [6.2261689342452256, 4.5200235099549895, 1.0471975511966036],
    [6.251168934245226, 4.563324780144211, 1.0471975511966036],
    [6.276168934245226, 4.606626050333433, 1.0471975511966036],
    [6.301168934245227, 4.649927320522655, 1.0471975511966036],
    [6.326168934245227, 4.693228590711876, 1.0471975511966036],
    [6.351168934245227, 4.736529860901098, 1.0471975511966036],
    [6.376168934245228, 4.77983113109032, 1.0471975511966036],
    [6.401168934245228, 4.823132401279541, 1.0471975511966036],
    [6.426168934245228, 4.866433671468763, 1.0471975511966036],
    [6.451168934245229, 4.909734941657985, 1.0471975511966036],
    [6.476168934245229, 4.953036211847206, 1.0471975511966036],
    [6.5011689342452295, 4.996337482036428, 1.0471975511966036],
    [6.52616893424523, 5.03963875222565, 1.0471975511966036],
    [6.55116893424523, 5.0829400224148715, 1.0471975511966036],
    [6.5761689342452305, 5.126241292604093, 1.0471975511966036],
    [6.601168934245231, 5.169542562793315, 1.0471975511966036],
    [6.626168934245231, 5.212843832982537, 1.0471975511966036],
    [6.651168934245232, 5.256145103171758, 1.0471975511966036],
    [6.676168934245232, 5.29944637336098, 1.0471975511966036],
    [6.701168934245232, 5.342747643550202, 1.0471975511966036],
    [6.726168934245233, 5.386048913739423, 1.0471975511966036],
    [6.751168934245233, 5.429350183928645, 1.0471975511966036],
    [6.776168934245233, 5.472651454117867, 1.0471975511966036],
    [6.801168934245234, 5.5159527243070885, 1.0471975511966036],
    [6.826168934245234, 5.55925399449631, 1.0471975511966036],
    [6.851168934245234, 5.602555264685532, 1.0471975511966036],
    [6.876168934245235, 5.645856534874754, 1.0471975511966036],
    [6.901168934245235, 5.689157805063975, 1.0471975511966036],
    [6.9261689342452355, 5.732459075253197, 1.0471975511966036],
    [6.951168934245236, 5.775760345442419, 1.0471975511966036],
    [6.976168934245236, 5.81906161563164, 1.0471975511966036],
    [7.001168934245237, 5.862362885820862, 1.0471975511966036],
    [7.026168934245237, 5.905664156010084, 1.0471975511966036],
    [7.051168934245237, 5.9489654261993055, 1.0471975511966036],
    [7.076168934245238, 5.992266696388527, 1.0471975511966036],
    [7.101168934245238, 6.035567966577749, 1.0471975511966036],
    [7.126168934245238, 6.0788692367669706, 1.0471975511966036],
    [7.151168934245239, 6.122170506956192, 1.0471975511966036],
    [7.176168934245239, 6.165471777145414, 1.0471975511966036],
    [7.201168934245239, 6.208773047334636, 1.0471975511966036],
    [7.22616893424524, 6.252074317523857, 1.0471975511966036],
    [7.25116893424524, 6.295375587713079, 1.0471975511966036],
    [7.2761689342452405, 6.338676857902301, 1.0471975511966036],
    [7.301168934245241, 6.381978128091522, 1.0471975511966036],
    [7.326168934245241, 6.425279398280744, 1.0471975511966036],
    [7.3511689342452415, 6.468580668469966, 1.0471975511966036],
    [7.376168934245242, 6.5118819386591875, 1.0471975511966036],
    [7.401168934245242, 6.555183208848409, 1.0471975511966036],
    [7.426168934245243, 6.598484479037631, 1.0471975511966036],
    [7.451168934245243, 6.641785749226853, 1.0471975511966036],
    [7.476168934245243, 6.685087019416074, 1.0471975511966036],
    [7.501168934245244, 6.728388289605296, 1.0471975511966036],
    [7.526168934245244, 6.771689559794518, 1.0471975511966036],
    [7.551168934245244, 6.814990829983739, 1.0471975511966036],
    [7.576168934245245, 6.858292100172961, 1.0471975511966036],
    [7.601168934245245, 6.901593370362183, 1.0471975511966036],
    [7.6261689342452454, 6.9448946405514045, 1.0471975511966036],
    [7.651168934245246, 6.988195910740626, 1.0471975511966036],
    [7.676168934245246, 7.031497180929848, 1.0471975511966036],
    [7.7011689342452465, 7.07479845111907, 1.0471975511966036],
    [7.726168934245247, 7.118099721308291, 1.0471975511966036],
    [7.751168934245247, 7.161400991497513, 1.0471975511966036],
    [7.776168934245248, 7.204702261686735, 1.0471975511966036],
    [7.801168934245248, 7.248003531875956, 1.0471975511966036],
    [7.826168934245248, 7.291304802065178, 1.0471975511966036],
    [7.851168934245249, 7.3346060722544, 1.0471975511966036],
    [7.876168934245249, 7.3779073424436215, 1.0471975511966036],
    [7.901168934245249, 7.421208612632843, 1.0471975511966036],
    [7.92616893424525, 7.464509882822065, 1.0471975511966036],
    [7.95116893424525, 7.5078111530112865, 1.0471975511966036],
    [7.97616893424525, 7.551112423200508, 1.0471975511966036],
    [8.00116893424525, 7.59441369338973, 1.0471975511966036],
    [8.026168934245248, 7.637714963578952, 1.0471975511966036],
    [8.051168934245247, 7.681016233768173, 1.0471975511966036],
    [8.076168934245246, 7.724317503957395, 1.0471975511966036],
    [8.101168934245244, 7.767618774146617, 1.0471975511966036],
    [8.126731398363143, 7.81058541335008, 0.9948376736367734],
    [8.154507530161638, 7.852155332264266, 0.9424777960769433],
    [8.184421197194167, 7.89221459063092, 0.8901179185171131],
    [8.216390408178668, 7.930653388807345, 0.8377580409572829],
    [8.250327537729984, 7.967366368719587, 0.7853981633974527],
    [8.286139566534953, 8.002252902641594, 0.7330382858376225],
    [8.32372833631189, 8.03521736900882, 0.6806784082777924],
    [8.362990818855595, 8.066169414510284, 0.6283185307179622],
    [8.40381939843052, 8.095024201740717, 0.575958653158132],
    [8.446102166738008, 8.121702641733991, 0.5235987755983018],
    [8.489068805941471, 8.14726510585189, 0.575958653158132],
    [8.530638724855658, 8.175041237650385, 0.6283185307179622],
    [8.570697983222313, 8.204954904682916, 0.6806784082777924],
    [8.609136781398739, 8.236924115667417, 0.7330382858376225],
    [8.64584976131098, 8.270861245218732, 0.7853981633974527],
    [8.680736295232988, 8.306673274023705, 0.8377580409572829],
    [8.713700761600213, 8.344262043800642, 0.8901179185171131],
    [8.744652807101678, 8.383524526344347, 0.9424777960769433],
    [8.77350759433211, 8.424353105919272, 0.9948376736367734],
    [8.800186034325383, 8.46663587422676, 1.0471975511966036],
    [8.824615003331873, 8.510256937137921, 1.0995574287564338],
    [8.846727543245569, 8.555096732351377, 1.151917306316264],
    [8.86646304513159, 8.601032357105224, 1.2042771838760942],
    [8.883767415351066, 8.647937905044957, 1.2566370614359244],
    [8.898593223828037, 8.69568481132404, 1.3089969389957545],
    [8.910899834051996, 8.74414220499121, 1.3613568165555847],
    [8.920653514459744, 8.793177267698654, 1.413716694115415],
    [8.92782753089125, 8.842655597747878, 1.466076571675245],
    [8.932402219866137, 8.892441578475413, 1.5184364492350753],
    [8.934365042479909, 8.942398749968667, 1.5707963267949054],
]

# Initialize environment
env = TwoDWorldDiffDrive(action_dt=PATH_DT)
env.reset(random_env=False)
env.add_ref_path(shortest_path)

controller = MPPI(env, action_mean=[0.25, 0], action_std=[0.25, math.radians(30)])
# debug
env.interactive_viz = True

# run controller
state = env.start_state
path = [state]
while True:
    action = controller.run(state)

    # debug
    state_samples = [state]
    best_actions = controller.prev_actions.tolist()  # debug
    new_state = env.sample_state_transition(state, action)[0]
    for future_action in best_actions:
        new_state = env.sample_state_transition(new_state, future_action)[0]
        state_samples.append(new_state)
    env.state_samples = state_samples

    next_state, reward, term, trunc, info = env.step(action)
    print(state, action, next_state, reward)

    # nearest_idx = env._get_nearest_waypoint_to_state(next_state)
    # print(shortest_path[nearest_idx])

    env.render()

    path.append(next_state)
    state = next_state

    if term or trunc:
        break

env.add_state_path(path)
env.render()