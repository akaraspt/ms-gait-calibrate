
class BodyLocation(object):
    """
    Enum for body location
    """
    Belt = 'belt'
    Chest = 'chest'
    Waist = 'waist'
    Waist_back = 'waist_back'
    Lower_back = 'lower_back'

    @classmethod
    def exist(self, body_loc):
        if (body_loc != self.Belt) and \
           (body_loc != self.Chest) and \
           (body_loc != self.Waist) and \
           (body_loc != self.Waist_back) and \
           (body_loc != self.Lower_back):
            return False

        return True


class Position(object):
    """
    Enum for position of the device
    """
    Cen = 'center'
    Left = 'left'
    Right = 'right'
    CenLeft = 'center_left'
    CenRight = 'center_right'

    @classmethod
    def exist(self, pos):
        if (pos != self.Cen) and \
           (pos != self.Left) and \
           (pos != self.Right) and \
           (pos != self.CenLeft) and \
           (pos != self.CenRight):
            return False

        return True